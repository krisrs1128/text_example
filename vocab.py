#!/usr/bin/env python


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2index_reduced = {}
        self.word2count = {}
        self.index2word = {"PAD_token": "PAD", "SOS_token": "SOS", "EOS_token": "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def replace_rare(self, min_count):
        for k, v in self.word2count.items():
            if v < min_count:
                self.word2index_reduced[k] = -1
            else:
                self.word2index_reduced[k] = v

    def reindex(self):
        reduced_ix = set(vocab.word2index_reduced.values())
        vocab_size = len(reduced_ix)
        reduced = dict(zip(reduced_ix, np.arange(vocab_size)))
        for k, v in self.word2index_reduced.items():
            self.word2index_reduced[k] = reduced[v]
