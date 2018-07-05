import pickle as pkl
import gzip
import os
import re
import sys
import numpy
import math
import random

from binary_tree import BinaryTree

def convert_ptb_to_tree(line):
    index = 0
    tree = None
    line = line.rstrip()

    stack = []
    parts = line.split()
    for p_i, p in enumerate(parts):
        try:
            # opening of a bracket, create a new node, take parent from top of stack
            if p == '(':
                if tree is None:
                    tree = BinaryTree(index)
                else:
                    add_descendant(tree, index, stack[-1])
                # add the newly created node to the stack and increment the index
                stack.append(index)
                index += 1
            # close of a bracket, pop node on top of the stack
            elif p == ')':
                stack.pop(-1)
            # otherwise, create a new node, take parent from top of stack, and set word
            else:
                add_descendant(tree, index, stack[-1])
                tree.set_word(index, p)
                index += 1
        except:
            continue
    return tree

def add_descendant(tree, index, parent_index):
    # add to the left first if possible, then to the right
    if tree.has_left_descendant_at_node(parent_index):
        if tree.has_right_descendant_at_node(parent_index):
            raise StopIteration
        else:
            tree.add_right_descendant(index, parent_index)
    else:
        tree.add_left_descendant(index, parent_index)


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, label,
                 dict,
                 batch_size=128,
                 n_words=-1,
                 maxlen=500,
                 shuffle=True):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.label = fopen(label, 'r')
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f,encoding='iso-8859-1')
        self.batch_size = batch_size
        self.n_words = n_words
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.end_of_data = False

        self.source_buffer = []
        self.target_buffer = []
        self.label_buffer = []
        self.k = batch_size * 20

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.label.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        label = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        assert len(self.source_buffer) == len(self.label_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                ll = self.label.readline()
                if ll == "":
                    break

                ss = convert_ptb_to_tree(ss)
                words_ss, left_mask_ss, right_mask_ss = ss.convert_to_sequence_and_masks(ss.root)
                words_ss = [self.dict[w] if w in self.dict else 1
                      for w in words_ss]
                if self.n_words > 0:
                    words_ss = [w if w < self.n_words else 1 for w in words_ss]
                ss = (words_ss, left_mask_ss, right_mask_ss)

                tt = convert_ptb_to_tree(tt)
                words_tt, left_mask_tt, right_mask_tt = tt.convert_to_sequence_and_masks(tt.root)
                words_tt = [self.dict[w] if w in self.dict else 1
                      for w in words_tt]
                if self.n_words > 0:
                    words_tt = [w if w < self.n_words else 1 for w in words_tt]
                tt = (words_tt, left_mask_tt, right_mask_tt)

                if len(words_ss) > self.maxlen or len(words_tt) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                self.label_buffer.append(ll.strip())

            if self.shuffle:
                # sort by target buffer
                tlen = numpy.array([len(t[0]) for t in self.target_buffer])
                tidx = tlen.argsort()
                # shuffle mini-batch
                tindex = []
                small_index = list(range(int(math.ceil(len(tidx)*1./self.batch_size))))
                random.shuffle(small_index)
                for i in small_index:
                    if (i+1)*self.batch_size > len(tidx):
                        tindex.extend(tidx[i*self.batch_size:])
                    else:
                        tindex.extend(tidx[i*self.batch_size:(i+1)*self.batch_size])

                tidx = tindex

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                _lbuf = [self.label_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                self.label_buffer = _lbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.label_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop(0)
                    tt = self.target_buffer.pop(0)
                    ll = self.label_buffer.pop(0)
                except IndexError:
                    break

                source.append(ss)
                target.append(tt)
                label.append(ll)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                        len(label) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0 or len(label) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, label

def prepare_data(group_x, group_y, labels):
    lengths_x = [len(s[0]) for s in group_x]
    lengths_y = [len(s[0]) for s in group_y]

    n_samples = len(group_x)
    maxlen_x = numpy.max(lengths_x)
    maxlen_y = numpy.max(lengths_y)

    x_seq = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y_seq = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    x_left_mask = numpy.zeros((maxlen_x, n_samples, maxlen_x)).astype('float32')
    x_right_mask = numpy.zeros((maxlen_x, n_samples, maxlen_x)).astype('float32')
    y_left_mask = numpy.zeros((maxlen_y, n_samples, maxlen_y)).astype('float32')
    y_right_mask = numpy.zeros((maxlen_y, n_samples, maxlen_y)).astype('float32')
    l = numpy.zeros((n_samples,)).astype('int64')

    for idx, [s_x, s_y, ll] in enumerate(zip(group_x, group_y, labels)):
        x_seq[-lengths_x[idx]:, idx] = s_x[0]
        x_mask[-lengths_x[idx]:, idx] = 1.
        x_left_mask[-lengths_x[idx]:, idx, -lengths_x[idx]:] = s_x[1]
        x_right_mask[-lengths_x[idx]:, idx, -lengths_x[idx]:] = s_x[2]
        y_seq[-lengths_y[idx]:, idx] = s_y[0]
        y_mask[-lengths_y[idx]:, idx] = 1.
        y_left_mask[-lengths_y[idx]:, idx, -lengths_y[idx]:] = s_y[1]
        y_right_mask[-lengths_y[idx]:, idx, -lengths_y[idx]:] = s_y[2]
        l[idx] = ll

    x = (x_seq, x_mask, x_left_mask, x_right_mask)
    y = (y_seq, y_mask, y_left_mask, y_right_mask)

    return x, y, l