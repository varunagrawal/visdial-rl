import json
import os
import os.path as osp
import pickle
import random
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import text
from utils.image import coco_name_format


def collate_fn(batch):
    # Sort batch (list) on question lengths for use in RNN pack_padded_sequence later
    batch.sort(key=lambda x: x["question_len"], reverse=True)
    return data.dataloader.default_collate(batch)


def get_dataloader(
        images,
        annotations,
        questions,
        pairs,
        split="train",
        maps=None,
        vocab=None,
        shuffle=False,
        batch_size=1,
        num_workers=0,
):
    return data.DataLoader(
        PulpDataset(images, annotations, questions,
                    pairs, split, vocab=vocab, maps=maps),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


class PulpDataset(Dataset):
    def __init__(
            self,
            images_dataset,
            annotations_file,
            questions_file,
            pairs_file,
            split="train",
            num_rounds=5,
            vocab=None,
            maps=None,
    ):
        """
            Initialize the dataset with split given by 'split', where
            split is taken from ['train', 'val', 'test']
        """
        self.images = torch.load(images_dataset)

        self.split = split
        self.num_rounds = num_rounds

        questions = json.load(open(questions_file)).get("questions")
        annotations = json.load(open(annotations_file)).get("annotations")

        self.prepare_dataset(annotations, questions, split, maps)

        # the above filters out some questions so we only use the questions we have
        valid_questions = {q["question_id"]: q for q in tqdm(self.data)}

        # this is an index to help get pairs quickly
        self.pairs = {}
        pairs = json.load(open(pairs_file))
        print("Filtering pairs")
        for q1, q2 in tqdm(pairs):
            # only add pairs that are in the dataset
            if q1 in valid_questions and q2 in valid_questions:
                self.pairs[q1] = q2
                self.pairs[q2] = q1

        # remove data points in self.data that don't have pairs
        self.data = [d for d in self.data if d["question_id"] in self.pairs]

        # self.ann_map = {a['question_id']: a for a in self.annotations}

        # generate the ques_map from the reduced dataset again
        self.ques_map = {q["question_id"]: q for q in tqdm(self.data)}

        self.im2ques_id = defaultdict(list)
        for q in self.ques_map.values():
            image_id = q["image_id"]
            q_id = q["question_id"]
            self.im2ques_id[image_id].append(q_id)

        self.question_index = {
            q["question_id"]: idx for idx, q in enumerate(self.data)
        }

    def prepare_dataset(self, annotations, questions, split="train", maps=None):
        self.data, self.vocab, \
            self.word_to_wid, self.wid_to_word, \
            self.ans_to_aid, self.aid_to_ans, \
            self.top_answers = process_vqa_dataset(questions, annotations,split, maps)

        # Add <START> and <END> to vocabulary
        word_count = len(self.word_to_wid)
        self.word_to_wid["<START>"] = word_count + 1
        self.word_to_wid["<END>"] = word_count + 2
        self.start_token = self.word_to_wid["<START>"]
        self.end_token = self.word_to_wid["<END>"]
        # Padding token is at index 0
        self.vocab_size = word_count + 3
        print("Vocab size with <START>, <END>: %d" % self.vocab_size)

        # We don't currently need captions
        # self.data['captions'] = self.process_caption(self.data['captions'],
        #                                              self.data['captions_len'])

        self.process_sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data item.
        # This is a single Image-Question-Answer datum.
        d = self.data[index]

        image_1 = d["image_id"]
        q1 = d["question_id"]

        # get all the question IDs for this image
        ques_ids = deepcopy(self.im2ques_id[image_1])

        # # remove q1 from ques_ids since we want to sample from the other questions
        # ques_ids.pop(ques_ids.index(q1))

        # sample a complementary image
        q2 = self.pairs[q1]
        image_2 = self.ques_map[q2]["image_id"]

        # we want 5 questions per round, so we sample from the complementary image if we don't
        if len(ques_ids) < self.num_rounds:
            # we want to sample from other questions not part of the current pair
            ques2_samples = deepcopy(self.im2ques_id[image_2])
            ques2_samples.pop(ques2_samples.index(q2))

            if len(ques2_samples) > 0:
                samples = np.random.choice(
                    range(0, len(ques2_samples)), size=self.num_rounds - len(ques_ids)
                )
                for s in samples:
                    ques_ids.append(ques2_samples[s])

            else:
                # only remove q1 if we have more questions to sample from
                if len(ques_ids) > 1:
                    ques_ids.pop(ques_ids.index(q1))

                samples = np.random.choice(
                    range(0, len(ques_ids)), size=self.num_rounds - 1
                ).tolist()
                ques_ids = [ques_ids[s] for s in samples]
                ques_ids.append(q1)

        elif len(ques_ids) > self.num_rounds:
            idx = ques_ids.index(q1)
            samples = np.random.choice(
                range(0, len(ques_ids)), size=self.num_rounds, replace=False
            ).tolist()
            if idx not in samples:
                samples[0] = idx

            ques_ids = [ques_ids[s] for s in samples]

        # assert len(ques_ids) == self.num_rounds, print(len(ques_ids), index)
        # print(index, image_1, q1, ques_ids)

        # Randomize the order of the questions
        random.shuffle(ques_ids)

        # Get the index of the discriminative question
        discriminant = ques_ids.index(q1)

        # flip a coin to decide if we should show the same two images or no
        different_image = np.random.rand() >= 0.5

        if not different_image:
            image_2 = image_1
            ques_2_ids = list(ques_ids)

        else:
            ques_2_ids = list(ques_ids)
            # Replace with the pair to get the appropriate answer
            ques_2_ids[ques_ids.index(q1)] = q2

        questions, ques_lens, answers_1, answers_2 = [], [], [], []
        # Get the question token, answer token tensors for each of the questions
        # and concatenate them
        for q in ques_ids:
            idx = self.question_index[q]
            questions.append(torch.from_numpy(
                self.data[idx]["question_wids"].astype(np.int64)))
            ques_lens.append(torch.LongTensor(self.data[idx]["question_length"]))
            answers_1.append(self.data[idx]["answer_id"])
        for q in ques_2_ids:
            answers_2.append(self.data[idx]["answer_id"])

        questions = torch.cat(questions).unsqueeze(0)
        question_lens = torch.cat(ques_lens).unsqueeze(0)
        answers_1 = torch.LongTensor(answers_1).unsqueeze(0)
        answers_2 = torch.LongTensor(answers_2).unsqueeze(0)

        img_feat_1 = self.normalize_feature(self.images[image_1].unsqueeze(0))
        img_feat_2 = self.normalize_feature(self.images[image_2].unsqueeze(0))

        d = {
            "image_1": img_feat_1,
            "image_2": img_feat_2,
            "questions": questions,
            "questions_lengths": question_lens,
            "answers_1": answers_1,
            "answers_2": answers_2,
            "discriminant": discriminant,
            "different": different_image,
        }

        return d

    def normalize_feature(self, img_feat):
        # normalize the image and embed it to the common dimension.
        norm = img_feat.norm(p=2, dim=1, keepdim=True).expand_as(img_feat)
        img_feat_norm = img_feat / norm.detach()
        return img_feat_norm

    def process_sequence(self):
        """
        Add <START> and <END> token to questions.
        """
        for d in self.data:
            wids = d["question_wids"]
            wids = np.hstack((self.word_to_wid["<START>"], wids, 0))
            wids[d["question_length"] + 1] = self.word_to_wid["<END>"]
            d["question_wids"] = wids
            # We add 1 for <START>
            d["question_length"] = wids.shape[0] + 1

    def process_caption(self, seq, seq_len):
        """
        Add <START> and <END> token to caption.
        Arguments:
            'seq'    : The captions for all the images
        """
        num_convs, max_cap_len = seq.size()
        new_size = torch.Size([num_convs, max_cap_len + 2])
        sequence = torch.LongTensor(new_size).fill_(0)

        # decodeIn begins with <START>
        sequence[:, 0] = self.word2ind["<START>"]
        end_token_id = self.word2ind["<END>"]

        for thId in range(num_convs):
            length = seq_len[thId]
            if length == 0:
                print("Warning: Skipping empty sequence at (%d)" % thId)
                continue

            sequence[thId, 1: length + 1] = seq[thId, :length]
            sequence[thId, length + 1] = end_token_id

        # Sequence length is number of tokens + 1
        seq_len = seq_len + 1
        seq = sequence

        return seq, seq_len


def process_vqa_dataset(questions, annotations, split,
                        maps=None, top_answer_limit=1000, 
                        max_length=26, year=2014):
    """
    Process the questions and annotations into a consolidated dataset.
    This is done only for the training set.
    :param questions_file:
    :param annotations_file:
    :param split: The dataset split.
    :param maps: Dict containing various mappings such as word_to_wid, wid_to_word, ans_to_aid and aid_to_ans.
    :param top_answer_limit:
    :param max_length: The maximum quetsion length. Taken from the VQA sample code.
    :param year: COCO Dataset release year.
    :return: The processed dataset ready to be used
    """
    cache_file = "vqa_{0}_dataset_cache.pickle".format(split)

    # Check if preprocessed cache exists. If yes, load it up, else preprocess the data
    if os.path.exists(cache_file):
        print("Found {0} set cache! Loading...".format(split))
        dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans, top_answers = pickle.load(
            open(cache_file, "rb")
        )

    else:
        # load up the dataset
        dataset = []
        for idx, q in enumerate(questions):
            d = dict()
            # question_id is unique
            d["question_id"] = q["question_id"]
            d["question"] = q["question"]
            d["image_id"] = q["image_id"]
            d["image_name"] = coco_name_format(q["image_id"], split, year)

            d["answer"] = annotations[idx]["multiple_choice_answer"]
            answers = []
            for ans in annotations[idx]["answers"]:
                answers.append(ans["answer"])
            d["answers_occurence"] = Counter(answers).most_common()

            d["question_type"] = annotations[idx]["question_type"]
            d["answer_type"] = annotations[idx]["answer_type"]

            dataset.append(d)

        if split == "train":
            # Get the top N answers so we can filter the dataset to only questions
            # with these answers
            top_answers = text.get_top_answers(dataset, top_answer_limit)
            dataset = text.filter_dataset(dataset, top_answers)

            # Process the questions
            dataset = text.preprocess_questions(dataset)

            vocab = text.get_vocabulary(dataset)

            # 0 is used for padding
            word_to_wid = {w: i + 1 for i, w in enumerate(vocab)}
            wid_to_word = {i + 1: w for i, w in enumerate(vocab)}
            ans_to_aid = {a: i for i, a in enumerate(top_answers)}
            aid_to_ans = {i: a for i, a in enumerate(top_answers)}

            dataset = text.encode_answers(dataset, ans_to_aid)

        else:  # split == "val":
            # filter the dataset to remove answers not in top_answers
            top_answers = maps['top_answers']
            dataset = text.filter_dataset(dataset, top_answers)

            # Process the questions
            dataset = text.preprocess_questions(dataset)

            vocab = maps["vocab"]

            word_to_wid = maps["word_to_wid"]
            wid_to_word = maps["wid_to_word"]
            ans_to_aid = maps["ans_to_aid"]
            aid_to_ans = maps["aid_to_ans"]

            dataset = text.remove_tail_words(dataset, vocab)
            dataset = text.encode_answers(dataset, ans_to_aid)

        dataset = text.encode_questions(dataset, word_to_wid, max_length)

        print("Caching the processed data")
        pickle.dump(
            [dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans, top_answers],
            open(cache_file, "wb+"),
        )

    return dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans, top_answers


if __name__ == "__main__":
    vqa_loader = get_dataloader(
        "image_embeddings/coco_train_vgg19_bn_fc7.pth",
        osp.expanduser("~/datasets/VQA2/v2_mscoco_train2014_annotations.json"),
        osp.expanduser(
            "~/datasets/VQA2/v2_OpenEnded_mscoco_train2014_questions.json"),
        osp.expanduser(
            "~/datasets/VQA2/v2_mscoco_train2014_complementary_pairs.json"),
        split="train",
    )

    print(len(vqa_loader))
    cnt = 0
    for d in tqdm(vqa_loader, total=len(vqa_loader)):
        cnt += 1

    print(cnt)
    print("done")
