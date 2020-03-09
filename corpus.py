import itertools as it
import os
import re

import regex as re
import toolz as tz
from bs4 import BeautifulSoup


def all_chars():
    alls = it.chain(
        [0x020],                  # space
        range(0x021, 0x080),      # ascii except space
        range(0x080, 0x100),      # latin1 supplement
        range(0x0400, 0x0500),    # russian
        range(0x2000, 0x2070),    # general punctuation
        range(0x2600, 0x2700),    # misc symbols
        range(0x1f300, 0x1f600),  # misc symbols and pictographs
        range(0x1f600, 0x1f650),  # smiles
        range(0x1f900, 0x1fa00),  # supplemental symbols and pictographs
        # ['•']                   # 0x2022 end of word
    )

    return {k: chr(k) for k in alls}

UNK = "࿕"
EOS = "∎"
ALL_CHARS = all_chars()
ALL_REV = {v: k for k, v in ALL_CHARS.items()}
ALL_KNOWN = {k: k for k in ALL_REV}


def find_tme_files(dir):
    return tz.thread_last(
        dir,
        os.listdir,
        (filter, lambda x: re.search(".*\.html$", x)),
        (map, lambda x: f"{dir}/{x}"),
        sorted,
    )


def tme_to_sentences(path):
    data = open(path).read()
    soup = BeautifulSoup(data, 'html.parser')
    divs = soup.find_all("div", "text")
    return [d.text.strip() for d in divs]


def split_to_tokens(s):
    # special kind of token for http links
    # (?:https?://[\p{L}0-9/\-_'.%=?&]+) # urls
    pat = re.compile("""
      [\p{L}0-9\-_'@]+                   # words
    | [^\s\p{L}\p{N}]+                   # non words
    | [\s]+                              # spaces
    | \s+(?!\S)                          # whatever non ws after ws
    """,
    re.VERBOSE)

    tokens = re.findall(pat, s)
    known_tokens = list()
    for t in tokens:
        known_tokens.append(
            "".join(ALL_KNOWN.get(s, UNK) for s in t)
        )
    return known_tokens


def tme_corpus_to_sentences(dir):
    files = find_tme_files(dir)
    for f in files:
        sent = tme_to_sentences(f)
        for v in sent:
            yield v


def tme_corpus_to_tokens(dir):
    files = find_tme_files(dir)
    for f in files:
        sent = tme_to_sentences(f)
        for v in sent:
            yield split_to_tokens(v)


def bible_corpus_to_sentences(path, drop_toc=True):
    lines = open(path).readlines()
    if drop_toc:
        lines = lines[1629:]
    for l in lines:
        s = l.strip()
        if s:
            yield s


def bible_corpus_to_tokens(path):
    lines = bible_corpus_to_sentences(path)
    for l in lines:
        yield split_to_tokens(l)


def all_to_corpus(tme_dir, bible_path, fname):
    with open(fname, 'w') as f:
        for s in tme_corpus_to_sentences(tme_dir):
            f.write(s)
            f.write(EOS)

        for s in bible_corpus_to_sentences(bible_path):
            f.write(s)
            f.write(EOS)


if __name__ == '__main__':
    import fire

    fire.Fire()
