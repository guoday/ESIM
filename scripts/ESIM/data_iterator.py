import pickle as pkl
import gzip
import numpy
import random
import math

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
                 shuffle=True):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.label = fopen(label, 'r')
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f,encoding='iso-8859-1')
        self.batch_size = batch_size
        self.n_words = n_words
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

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
                self.label_buffer.append(ll.strip())

            if self.shuffle:
                # sort by target buffer
                tlen = numpy.array([len(t) for t in self.target_buffer])
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
                except IndexError:
                    break

                ss.insert(0, '_BOS_')
                ss.append('_EOS_')
                ss = [self.dict[w] if w in self.dict else 1
                      for w in ss]
                if self.n_words > 0:
                    ss = [w if w < self.n_words else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop(0)
                tt.insert(0, '_BOS_')
                tt.append('_EOS_')
                tt = [self.dict[w] if w in self.dict else 1
                      for w in tt]
                if self.n_words > 0:
                    tt = [w if w < self.n_words else 1 for w in tt]

                # read label 
                ll = self.label_buffer.pop(0)

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



def prepare_data(seqs_x, seqs_y, labels, worddicts_r, alphabet,maxlen=None):
    alphabet_r=dict()
    for i in range(len(alphabet)):
        alphabet_r[alphabet[i]]=i


    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        new_labels = []
        for l_x, s_x, l_y, s_y, l in zip(lengths_x, seqs_x, lengths_y, seqs_y, labels):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_labels.append(l)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        labels = new_labels

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None

    max_char_len_x = 0
    max_char_len_y = 0
    seqs_x_char = []
    l_seqs_x_char = []
    seqs_y_char = []
    l_seqs_y_char = []

    for idx, [s_x, s_y, s_l] in enumerate(zip(seqs_x, seqs_y, labels)):
        temp_seqs_x_char = []
        temp_l_seqs_x_char = []
        temp_seqs_y_char = []
        temp_l_seqs_y_char = []
        for w_x in s_x:
            word = worddicts_r[w_x]
            word_list = list(word)
            l_word_list = len(word_list)
            temp_seqs_x_char.append(word_list)
            temp_l_seqs_x_char.append(l_word_list)
            if l_word_list >= max_char_len_x:
                max_char_len_x = l_word_list
        for w_y in s_y:
            word = worddicts_r[w_y]
            word_list = list(word)
            l_word_list = len(word_list)
            temp_seqs_y_char.append(word_list)
            temp_l_seqs_y_char.append(l_word_list)
            if l_word_list >= max_char_len_y:
                max_char_len_y = l_word_list

        seqs_x_char.append(temp_seqs_x_char)
        l_seqs_x_char.append(temp_l_seqs_x_char)
        seqs_y_char.append(temp_seqs_y_char)
        l_seqs_y_char.append(temp_l_seqs_y_char)


    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x)
    maxlen_y = numpy.max(lengths_y)

    x = numpy.zeros((n_samples,maxlen_x)).astype('int32')
    y = numpy.zeros((n_samples,maxlen_y)).astype('int32')
    x_mask = numpy.zeros((n_samples,maxlen_x)).astype('float32')
    y_mask = numpy.zeros((n_samples,maxlen_y)).astype('float32')
    l = numpy.zeros((n_samples,)).astype('int32')
    char_x1 = numpy.zeros((n_samples,maxlen_x, max_char_len_x)).astype('int32')
    char_x1_mask = numpy.zeros((n_samples,maxlen_x, max_char_len_x)).astype('float32')
    char_x2 = numpy.zeros((n_samples,maxlen_y, max_char_len_y)).astype('int32')
    char_x2_mask = numpy.zeros((n_samples,maxlen_y, max_char_len_y)).astype('float32')

    for idx, [s_x, s_y, ll] in enumerate(zip(seqs_x, seqs_y, labels)):
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.
        y[idx,:lengths_y[idx]] = s_y
        y_mask[idx,:lengths_y[idx]] = 1.
        l[idx] = ll

        for j in range(0, lengths_x[idx]):
            char_x1[idx,j, :l_seqs_x_char[idx][j]] = [alphabet_r[char] if char in alphabet else 0 for char in seqs_x_char[idx][j]]
            char_x1_mask[idx,j, :l_seqs_x_char[idx][j]] = 1.
        for j in range(0, lengths_y[idx]):
            char_x2[idx, j, :l_seqs_y_char[idx][j]] = [alphabet_r[char] if char in alphabet else 0 for char in seqs_y_char[idx][j]]
            char_x2_mask[idx, j, :l_seqs_y_char[idx][j]] = 1.

    return x, x_mask, char_x1, char_x1_mask, y, y_mask, char_x2, char_x2_mask,lengths_x,lengths_y,l
