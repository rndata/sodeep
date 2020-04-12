import itertools as it

import numpy as np
import tensorflow as tf
import toolz as tz

from .generic import (
    ALL_CHARS,
    EOS,
    UNK,
    bible_corpus_to_sentences,
    tme_corpus_to_sentences,
)


class SimpleDecoder:

    @classmethod
    def from_chars(cls, all_chars, *, unk, eos):
        alls = set(it.chain([eos, unk], all_chars, ))

        chr2ix = dict(zip(alls, it.count(1)))
        ix2chr = {v: k for k, v in chr2ix.items()}

        return cls(chr2ix, ix2chr, unk, eos)

    def __init__(self, chr2ix, ix2chr, unk, eos):
        self.eos = eos
        self.unk = unk

        self.chr2ix = chr2ix
        self.ix2chr = ix2chr

        self.eos_ix = self.chr2ix[self.eos]
        self.unk_ix = self.chr2ix[self.unk]

    def encode_raw(self, sent):
        return self.encode(sent + self.eos)

    def encode(self, sent):
        return [self.chr2ix.get(c, self.unk_ix) for c in sent]

    def decode(self, enc):
        return "".join([self.ix2chr[c] for c in enc if c != 0])

    def to_features(self):
        chrs = "".join(self.chr2ix)
        btes = chrs.encode()
        ixes = list(self.chr2ix.values())

        return tf.train.Features(feature={
            "chrs": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[btes])),
            "ixes": tf.train.Feature(
                int64_list=tf.train.Int64List(value=ixes)),
        })


def examples(encoder, gen):

    exes = tz.thread_last(
        gen,
        (tz.sliding_window, 3),
        (map, lambda x: [encoder.encode_raw(i) for i in x]),
        (map, tz.compose(list, tz.concat)),
        (map, np.array)
    )

    return exes


def write_examples(fname, exes):
    with tf.io.TFRecordWriter(fname) as writer:
        for e in exes:
            writer.write(e.SerializeToString())


def default_encoder():
    return SimpleDecoder.from_chars(ALL_CHARS.values(), unk=UNK, eos=EOS)


def examples_gens(path):

    enc = default_encoder()
    bo = examples(
        enc,
        tme_corpus_to_sentences(f"{path}/blue-oyster/"),
    )
    lb = examples(
        enc,
        tme_corpus_to_sentences(f"{path}/lenta4-bot/"),
    )
    bb = examples(
        enc,
        bible_corpus_to_sentences(f"{path}/bible.txt"),
    )

    return bo, lb, bb


def default_ds(path):

    def data():
        return filter(lambda x: len(x)<300, it.chain(*examples_gens(path)))

    ds = tf.data.Dataset.from_generator(
        data,
        output_types=tf.int64,
    )

    x = ds.map(lambda x: x[:-1]).padded_batch(64, [None], drop_remainder=True)
    y = ds.map(lambda x: x[1:]).padded_batch(64, [None], drop_remainder=True)
    ds = tf.data.Dataset.zip((x,y))

    # ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    # ds = ds.shuffle(10000)

    return ds
