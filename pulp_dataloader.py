import json
import os
import os.path as osp
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.utils.data as data
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

from utils import text
from utils.image import coco_name_format


def collate_fn(batch):
    # Sort batch (list) on question lengths for use in RNN pack_padded_sequence later
    batch.sort(key=lambda x: x['question_len'], reverse=True)
    return data.dataloader.default_collate(batch)


def get_dataloader(images, annotations, questions, pairs, split="train",
                   maps=None, vocab=None, shuffle=False, batch_size=1, num_workers=0):
    return data.DataLoader(PulpDataset(images, annotations, questions, pairs, split,
                                       vocab=vocab, maps=maps),
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle)


class PulpDataset(Dataset):
    def __init__(self, images_dataset, annotations_file, questions_file, pairs_file,
                 split="train", num_rounds=5, vocab=None, maps=None):
        """
            Initialize the dataset with split given by 'split', where
            split is taken from ['train', 'val', 'test']
        """
        self.images = torch.load(images_dataset)

        self.split = split

        pairs = json.load(open(pairs_file))
        # this is an index to help get pairs quickly
        self.pairs = {}
        for q1, q2 in pairs:
            self.pairs[q1] = q2
            self.pairs[q2] = q1

        self.questions = json.load(open(questions_file)).get("questions")
        self.annotations = json.load(open(annotations_file)).get("annotations")

        self.ques_map = {q['question_id']: q for q in self.questions}
        # self.ann_map = {a['question_id']: a for a in self.annotations}

        self.im2ques_id = defaultdict(list)
        for k, v in self.ques_map.items():
            image_id = v['image_id']
            self.im2ques_id[image_id].append(k)

        self.num_rounds = num_rounds

        self.prepare_dataset(self.annotations, self.questions, split, maps)

        self.question_index = {q['question_id']
            : idx for idx, q in enumerate(self.data)}

    def prepare_dataset(self, annotations, questions, split="train", maps=None):
        self.data, self.vocab, \
            self.word_to_wid, self.wid_to_word, \
            self.ans_to_aid, self.aid_to_ans = \
            process_vqa_dataset(questions, annotations, split, maps)

        # Add <START> and <END> to vocabulary
        word_count = len(self.word_to_wid)
        self.word_to_wid['<START>'] = word_count + 1
        self.word_to_wid['<END>'] = word_count + 2
        self.start_token = self.word_to_wid['<START>']
        self.end_token = self.word_to_wid['<END>']
        # Padding token is at index 0
        self.vocab_size = word_count + 3
        print('Vocab size with <START>, <END>: %d' % self.vocab_size)

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

        image_1 = d['image_id']
        q1 = d['question_id']

        # get all the question IDs for this image
        ques_ids = self.im2ques_id[image_1]
        # # remove q1 from ques_ids since we want to sample from the other questions
        # ques_ids.pop(ques_ids.index(q1))

        # sample a complementary image
        q2 = self.pairs[q1]
        image_2 = self.ques_map[q2]['image_id']

        # we want at least 5 questions per round, so we sample from the complementary image if we don't
        if len(ques_ids) < self.num_rounds:
            # we want to sample from other questions not part of the current pair
            ques2_samples = self.im2ques_id[image_2]
            ques2_samples.pop(ques2_samples.index(q2))
            samples = np.random.randint(0, len(ques2_samples),
                                        size=self.num_rounds-len(ques_ids))
            for s in samples:
                ques_ids.append(ques2_samples[s])

        assert len(ques_ids) == self.num_rounds

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

        questions, answers_1, answers_2 = [], [], []
        # Get the question token, answer token tensors for each of the questions and concatenate them
        for q in ques_ids:
            index = self.question_index[q]
            questions.append(torch.from_numpy(
                self.data[index]['question_wids'].astype(np.int64)))
            #answers_1.append(torch.from_numpy(self.data[index]['answer_id']))

        #for q in ques_2_ids:
        #    answers_2.append(torch.from_numpy(self.data[index]['answer_id']))

        questions = torch.cat(questions).unsqueeze(0)
        answers_1 = torch.tensor([1])#torch.cat(answers_1).unsqueeze(0)

        answers_2 = torch.tensor([1])#torch.cat(answers_2).unsqueeze(0)

        d = {
            "image_1": self.images[image_1],
            "image_2": self.images[image_2],
            "questions": questions,
            "answers_1": answers_1,
            "answers_2": answers_2,
            "discriminant": discriminant,
            "different": different_image
        }

        return d

    def process_sequence(self):
        """
        Add <START> and <END> token to questions.
        """
        for d in self.data:
            wids = d['question_wids']
            wids = np.hstack((self.word_to_wid['<START>'], wids, 0))
            wids[d['question_length']+1] = self.word_to_wid['<END>']
            d['question_wids'] = wids
            # We add 1 for <START>
            d['question_length'] = wids.shape[0] + 1

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
        sequence[:, 0] = self.word2ind['<START>']
        end_token_id = self.word2ind['<END>']

        for thId in range(num_convs):
            length = seq_len[thId]
            if length == 0:
                print('Warning: Skipping empty sequence at (%d)' % thId)
                continue

            sequence[thId, 1:length + 1] = seq[thId, :length]
            sequence[thId, length + 1] = end_token_id

        # Sequence length is number of tokens + 1
        seq_len = seq_len + 1
        seq = sequence

        return seq, seq_len


def process_vqa_dataset(questions, annotations, split, maps=None,
                        top_answer_limit=1000, max_length=26, year=2014):
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
        dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans = pickle.load(
            open(cache_file, 'rb'))

    else:
        # load up the dataset
        dataset = []
        for idx, q in enumerate(questions):
            d = dict()
            d["question_id"] = q["question_id"]
            d["question"] = q["question"]
            d["image_id"] = q["image_id"]
            d["image_name"] = coco_name_format(q["image_id"], split, year)

            d["answer"] = annotations[idx]["multiple_choice_answer"]
            answers = []
            for ans in annotations[idx]['answers']:
                answers.append(ans['answer'])
            d['answers_occurence'] = Counter(answers).most_common()

            d["question_type"] = annotations[idx]["question_type"]
            d["answer_type"] = annotations[idx]["answer_type"]

            dataset.append(d)

        if split == "train":
            # Get the top N answers so we can filter the dataset to only questions with these answers
            top_answers = text.get_top_answers(dataset, top_answer_limit)
            dataset = text.filter_dataset(dataset, top_answers)

            # Process the questions
            dataset = text.preprocess_questions(dataset)

            vocab = text.get_vocabulary(dataset)

            # 0 is used for padding
            word_to_wid = {w: i+1 for i, w in enumerate(vocab)}
            wid_to_word = {i+1: w for i, w in enumerate(vocab)}
            ans_to_aid = {a: i for i, a in enumerate(top_answers)}
            aid_to_ans = {i: a for i, a in enumerate(top_answers)}

            dataset = text.encode_answers(dataset, ans_to_aid)

        else:  # split == "val":
            # Process the questions
            dataset = text.preprocess_questions(dataset)

            vocab = maps["vocab"]

            word_to_wid = maps["word_to_wid"]
            wid_to_word = maps["wid_to_word"]
            ans_to_aid = maps["ans_to_aid"]
            aid_to_ans = maps["aid_to_ans"]

            dataset = text.remove_tail_words(dataset, vocab)

        dataset = text.encode_questions(dataset, word_to_wid, max_length)

        print("Caching the processed data")
        pickle.dump([dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans],
                    open(cache_file, 'wb+'))

    return dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans


if __name__ == "__main__":
    vqa_loader = get_dataloader("image_embeddings/coco_val_vgg19_bn_fc7.pth",
                                osp.expanduser(
                                    "~/datasets/VQA2/v2_mscoco_train2014_annotations.json"),
                                osp.expanduser(
                                    "~/datasets/VQA2/v2_OpenEnded_mscoco_train2014_questions.json"),
                                osp.expanduser(
                                    "~/datasets/VQA2/v2_mscoco_train2014_complementary_pairs.json"),
                                split='train')

    for d in vqa_loader:
        print(d)
