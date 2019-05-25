import numpy as np
import pandas as pd
import pickle
import torch
import itertools
import re
import sys
import os

from lime.lime_text import LimeTextExplainer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam, SGD

from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def make_dictionary(sentences, vocabulary_size=None, initial_words=['<UNK>', '<PAD>', '<SOS>', '<EOS>']):
    """sentences : list of list"""

    counter = Counter()
    for words in sentences:
        counter.update(words)

    if vocabulary_size is None:
        vocabulary_size = len(counter.keys())

    vocab_words = counter.most_common(vocabulary_size)

    for initial_word in initial_words:
        vocab_words.insert(0, (initial_word, 0))

    word2idx = {word: idx for idx, (word, count) in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def process_sentences(sentences, word2idx, sentence_length=20, padding='<PAD>'):
    """sentences : list of list
    Only paddding. No SOS or EOS
    """

    sentences_processed = []
    for sentence in sentences:
        if len(sentence) > sentence_length:
            fixed_sentence = sentence[:sentence_length]
        else:
            fixed_sentence = sentence + [padding] * (sentence_length - len(sentence))

        sentence_idx = [word2idx[word] if word in word2idx.keys() else word2idx['<UNK>'] for word in fixed_sentence]

        sentences_processed.append(sentence_idx)

    return sentences_processed


def make_mask(sentences, sentence_length=20):
    masks = []
    for sentence in sentences:
        words_count = len(sentence[:sentence_length])
        sentence_mask = np.concatenate([np.ones(words_count - 1), np.ones(1), np.zeros(sentence_length - words_count)])
        masks.append(sentence_mask)

    mask = np.array(masks)
    return mask


with open(os.path.join(BASE_DIR, 'params.pkl'), 'rb') as pkl:
    params = pickle.load(pkl)
word2idx = params['word2idx']
idx2word = params['idx2word']
SENTENCE_LENGTH = params['sentence_length']


# # Build model

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_size, c_size, kernel_num, kernel_sizes):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=0
        )
        self.conv_list = [
            nn.Conv1d(embed_size, kernel_num, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ]
        self.convs = nn.ModuleList(self.conv_list)

        self.maxpools = nn.ModuleList([
            nn.MaxPool1d(kernel_size)
            for kernel_size in kernel_sizes
        ])

        self.linear = nn.Linear(2200, c_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)

        pools = []
        for conv, maxpool in zip(self.convs, self.maxpools):
            feature_map = conv(embedded)
            pooled = maxpool(feature_map)
            pools.append(pooled)

        conv_concat = torch.cat(pools, dim=-1).view(batch_size, -1)
        conv_concat = self.dropout(conv_concat)
        logits = self.linear(conv_concat)
        return self.softmax(logits)


D = Discriminator(
    vocab_size=len(word2idx),
    embed_size=128,
    c_size=2,
    kernel_num=100,
    kernel_sizes=[2, 3, 4, 5]
)

D.load_state_dict(torch.load(os.path.join(BASE_DIR, 'D_180115.pth'), map_location='cpu'))

# evaluation for fixed dropout
D.eval()


def clean(s):
    ss = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]+', '', s)
    ssss = ''.join(ch if len(list(grouper)) == 1 else ch * 2 for ch, grouper in itertools.groupby(ss))
    return ssss


def do_inference(raw_sentences, print_clean=False):
    clean_sentences = [clean(s) for s in raw_sentences]
    sentences = [list(''.join(clean_sentence.split())) for clean_sentence in clean_sentences]
    infer_sentences_processed = process_sentences(sentences, word2idx, sentence_length=SENTENCE_LENGTH)
    data = torch.LongTensor(infer_sentences_processed)
    log_probs = D(Variable(data))
    probs = log_probs.exp()
    return probs


def inference_one(string):
    # predict single sentence
    res = do_inference([string])
    return float(int(res[0][1] * 100) / 100)


def spacing_example(example):
    # separate sentence word for limer
    length = len(example.split())
    if length < 2:
        spaced = ' '.join([c for c in example.replace(' ', '')])
    else:
        spaced = example
    return spaced


def limer(example):
    # show in lime graph
    # TODO: ext -> html로 return
    # note가 아닌 html API 찾기
    explainer = LimeTextExplainer()
    exp = explainer.explain_instance(spacing_example(example), lambda s: do_inference(s, True).detach().numpy(),
                                     top_labels=1)
    exp.show_in_notebook()


if __name__ == '__main__':
    sentence = sys.argv[1]
    slang_accuracy = inference_one(sentence)
    print(f'''
Your sentence: {sentence}
Slang accuracy: {slang_accuracy}
''')
