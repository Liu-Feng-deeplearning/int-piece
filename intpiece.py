# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
from functools import partial
from itertools import chain
from multiprocessing import Pool, Queue

import ahocorasick
import numpy as np
from tqdm import tqdm, trange

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger_x = logging

try:
    import faster

    USE_C = True
    logger_x.info("Using faster.pyx for fast processing.")
except Exception as e:
    logger_x.info("Fails to load faster.pyx, and use py func instead.")
    USE_C = False


def normalize(text):
    return text


class Trainer:
    """A novel unsupervised training algorithm for Unigram
    Reference: https://kexue.fm/archives/3956
    """

    def __init__(
        self,
        max_seq_id=16383,
        order=6,
        max_vocab_size=10000,
        max_piece_length=36,
        min_count=2,
    ):
        self.order = order
        self.max_piece_length = max_piece_length
        self.min_count = min_count
        self.max_seq_id = max_seq_id
        if isinstance(max_vocab_size, list):
            self.max_vocab_size = sorted(max_vocab_size)[::-1]
        else:
            self.max_vocab_size = [max_vocab_size]

    def count_ngrams(self, texts):
        ngrams = [{} for i in range(self.order + 1)]
        text = list(texts)
        for i in range(len(text)):
            for j in range(self.order + 1):
                k = text[i:i + j]
                k = tuple(k)
                ngrams[j][k] = ngrams[j].get(k, 0) + 1
        return ngrams

    def prune_ngrams(self, ngrams):
        for i in range(self.max_seq_id):
            p = tuple([i])
            if p not in ngrams[1]:
                ngrams[1][p] = 1
                ngrams[0][tuple([])] += 1

        for i in trange(len(ngrams) - 1, -1, -1, desc='Prune Ngrams', ncols=0):
            ngrams[i] = {
                k: np.log(v)
                for k, v in ngrams[i].items()
                if len(k) == i and v >= (self.min_count if i > 1 else 0)
            }
            if i < len(ngrams) - 1:
                ngrams[i + 1] = {
                    k: v - ngrams[i][k[:i]]
                    for k, v in ngrams[i + 1].items()
                }
        return ngrams

    @property
    def trans(self):
        if not hasattr(self, '_trans'):
            self._trans = np.full((self.order, self.order), -np.inf)
            for i in range(self.order):
                self._trans[i, 0] = 0
                self._trans[i, min(i + 1, self.order - 1)] = 0
        return self._trans

    def _tokenize(self, text):
        # Nodes
        nodes = np.full((len(text), self.order), -np.inf)
        for j in range(self.order):
            for i in range(j, len(text)):
                nodes[i, j] = self.ngrams[j + 1].get(tuple(text[i - j:i + 1]),
                                                     -np.inf)

        # Viterbi
        routes = np.zeros((len(text) - 1, self.order), dtype='int32')
        for i in range(1, len(nodes)):
            scores = nodes[i - 1][:, None] + self.trans + nodes[i]
            routes[i - 1] = scores.argmax(0)
            nodes[i] = scores.max(0)

        # Output
        opt_route = [nodes[-1].argmax()]
        for i in range(1, len(nodes)):
            opt_route.append(routes[-i][opt_route[-1]])
        opt_route = np.array(opt_route[::-1])
        opt_route = np.append(np.where(opt_route == 0)[0], len(nodes))
        return [text[s:e] for s, e in zip(opt_route, opt_route[1:])]

    def count_pieces(self, texts):
        pieces = {}
        text = list(texts)
        for p in self._tokenize(text):
            p = tuple(p)
            pieces[p] = pieces.get(p, 0) + 1
        return pieces

    def split_pieces(self, keep, drop):
        tokenizer, counter = Tokenizer(self.dump(keep)), {}
        for k, v in drop:
            for p in tokenizer._tokenize(k):
                p = tuple(p)
                counter[p] = counter.get(p, 0) + v
        return counter

    def prune_pieces(self, pieces, workers=1, batch_size=1000):
        desc = 'Prune Pieces'
        split_pieces = partial(
            self.psplit_pieces, workers=workers, batch_size=batch_size
        ) if workers > 1 else self.split_pieces

        # Complete all bytes
        for i in range(self.max_seq_id):
            p = tuple([i])
            if p not in pieces:
                pieces[p] = 1

        keep_pieces, drop_pieces = {}, {}
        for k, v in pieces.items():
            if len(k) == 1 or (
                len(k) <= self.max_piece_length and v >= self.min_count
            ):
                keep_pieces[k] = v
            else:
                drop_pieces[k] = v
        drop_pieces = tqdm(drop_pieces.items(), desc=desc, ncols=0)
        for k, v in split_pieces(keep_pieces, drop_pieces).items():
            keep_pieces[k] += v
        # Prune wasted pieces
        while True:
            len_keep_pieces = len(keep_pieces)
            drop_pieces = tqdm(keep_pieces.items(), desc=desc, ncols=0)
            keep_pieces = split_pieces(keep_pieces, drop_pieces)
            if len_keep_pieces == len(keep_pieces):
                break
        # Prune by max_vocab_size
        final_pieces = []
        for max_vocab_size in self.max_vocab_size:
            if len(keep_pieces) <= max_vocab_size - 3:
                final_pieces.append(keep_pieces)
                continue
            pieces = sorted(
                keep_pieces.items(),
                key=lambda t: (len(t[0]) > 1, -t[1], -len(t[0]), t[0])
            )
            keep_pieces = dict(pieces[:max_vocab_size - 3])
            drop_pieces = tqdm(pieces[max_vocab_size - 3:], desc=desc, ncols=0)
            for k, v in split_pieces(keep_pieces, drop_pieces).items():
                keep_pieces[k] += v
            # Prune wasted pieces
            while True:
                len_keep_pieces = len(keep_pieces)
                drop_pieces = tqdm(keep_pieces.items(), desc=desc, ncols=0)
                keep_pieces = split_pieces(keep_pieces, drop_pieces)
                if len_keep_pieces == len(keep_pieces):
                    break
            final_pieces.append(keep_pieces)
        # Output
        return final_pieces

    def norm(self, texts):
        for text in texts:
            for t in normalize(text):
                yield t

    def train(self, texts, workers=1, batch_size=1000):
        if workers > 1:
            texts1 = self.norm(tqdm(texts, desc='Count Ngrams'))
            self.ngrams = self.pcount_ngrams(texts1, workers, batch_size)
            self.ngrams = self.prune_ngrams(self.ngrams)
            texts2 = self.norm(tqdm(texts, desc='Count Pieces'))
            self.pieces = self.pcount_pieces(texts2, workers, batch_size)
            self.pieces = self.prune_pieces(self.pieces, workers, batch_size)
        else:
            texts1 = self.norm(tqdm(texts, desc='Count Ngrams'))
            self.ngrams = self.count_ngrams(texts1)
            self.ngrams = self.prune_ngrams(self.ngrams)
            texts2 = self.norm(tqdm(texts, desc='Count Pieces'))
            self.pieces = self.count_pieces(texts2)
            self.pieces = self.prune_pieces(self.pieces)

    def dump(self, pieces):
        pieces = sorted(pieces.items(), key=lambda t: (len(t[0]), t[0]))
        # pieces = pieces.items()
        return {
            k: [i + 3, k, v]
            for i, (k, v) in enumerate(pieces)
        }

    def save(self, path):
        if len(self.pieces) == 1:
            paths = [path]
        else:
            paths = ['%s.%s' % (path, size) for size in self.max_vocab_size]
        for pieces, path in zip(self.pieces, paths):
            dump_dict = {}
            for k, v in self.dump(pieces).items():
                k_str = " ".join([str(x) for x in list(k)])
                dump_dict[k_str] = v
            json.dump(
                dump_dict,
                open(path, 'w'),
                indent=4,
                ensure_ascii=False
            )

    def pcount(self, inputs, count, merge, init, desc, workers, batch_size):
        def worker_func(in_queue, out_queue):
            counter = init()
            while True:
                inputs = in_queue.get()
                if inputs is None:
                    break
                merge(counter, count(inputs))
            out_queue.put(counter)

        # Count
        in_queue, out_queue = Queue(workers + 1), Queue()
        pool = Pool(workers, worker_func, (in_queue, out_queue))
        batch = []
        for input in inputs:
            batch.append(input)
            if len(batch) == batch_size:
                in_queue.put(batch)
                batch = []
        if batch:
            in_queue.put(batch)
        for i in range(workers):
            in_queue.put(None)
        # Merge
        counter = init()
        for _ in trange(workers, desc=desc, ncols=0):
            merge(counter, out_queue.get())
        pool.terminate()
        return counter

    def pcount_ngrams(self, texts, workers=1, batch_size=1000):
        def merge(ngrams1, ngrams2):
            for i, G in enumerate(ngrams2):
                for k, v in G.items():
                    ngrams1[i][k] = ngrams1[i].get(k, 0) + v

        init = lambda: [{} for i in range(self.order + 1)]
        return self.pcount(
            texts, self.count_ngrams, merge, init, 'Merge Ngrams', workers,
            batch_size
        )

    def psplit_pieces(self, keep, drop, workers=1, batch_size=1000):
        def merge(pieces1, pieces2):
            for k, v in pieces2.items():
                pieces1[k] = pieces1.get(k, 0) + v

        split_pieces = lambda drop: self.split_pieces(keep, drop)
        return self.pcount(
            drop, split_pieces, merge, dict, 'Merge Pieces', workers,
            batch_size * 10
        )

    def pcount_pieces(self, texts, workers=1, batch_size=1000):
        def merge(pieces1, pieces2):
            for k, v in pieces2.items():
                pieces1[k] = pieces1.get(k, 0) + v

        return self.pcount(
            texts, self.count_pieces, merge, dict, 'Merge Pieces', workers,
            batch_size // 10
        )


class Tokenizer:
    """Unigram tokenizer with Aho-Corasick automaton"""

    def __init__(self, pieces):
        if isinstance(pieces, str):
            pieces = json.load(open(pieces))
            new_pieces = dict()
            for k_str, v in pieces.items():
                k = [int(x) for x in k_str.split()]
                k = tuple(k)
                new_pieces[k] = v
            pieces = new_pieces

        pieces = {k: v for k, v in pieces.items()}
        self._pieces = {k: v[-1] for k, v in pieces.items()}
        self._piece2id = {k: v[0] for k, v in pieces.items()}
        for i, k in enumerate(['<pad>', '<bos>', '<eos>']):
            self._piece2id[k] = i
        self._id2piece = {v: k for k, v in self._piece2id.items()}
        self.vocab_size = len(self._pieces) + 3
        # Aho-Corasick automaton
        log_total = np.log(sum(self._pieces.values()))
        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY,
                                                ahocorasick.KEY_SEQUENCE)
        for k, v in self._pieces.items():
            self._automaton.add_word(k, (len(k), np.log(v) - log_total))
        self._automaton.make_automaton()
        return

    def _tokenize(self, text, alpha=-1):
        if USE_C:
            text = list(text)
            return faster._tokenize(self, text, alpha)
        else:
            return self._tokenize_with_py(text)

    def _tokenize_with_py(self, text: list):
        scores = [0] + [-100000000] * len(text)
        routes = list(range(len(text) + 1))
        tokens = []
        for e, (k, v) in self._automaton.iter(tuple(text)):
            s, e = e - k + 1, e + 1
            score = scores[s] + v
            if score > scores[e]:
                scores[e], routes[e] = score, s

        while text:
            s = routes[e]
            tokens.append(tuple(text[s:e]))
            text, e = text[:s], s
        return tokens[::-1]

    def tokenize(self, text):
        text = normalize(text)
        pieces = self._tokenize(text)
        return list(pieces)

    def piece_to_id(self, p):
        return self._piece2id[p]

    def id_to_piece(self, i):
        return self._id2piece[i]

    def pieces_to_ids(self, pieces):
        return [self._piece2id[p] for p in pieces]

    def ids_to_pieces(self, ids):
        return [self._id2piece[i] for i in ids]

    def encode(self, text, add_bos=False, add_eos=False, alpha=-1):
        def generator():
            if add_bos:
                yield 1
            for p in self.tokenize(text):
                yield self._piece2id[p]
            if add_eos:
                yield 2

        return list(generator())

    def decode(self, ids):
        pieces = [self._id2piece[i] for i in ids if i > 2]
        return list(chain(*pieces))


class Corpus:
    def __init__(self, data_path, use_seq_length=-1):
        self.data_path = data_path
        self.use_seq_length = use_seq_length
        return

    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                tokens = json.loads(line)['token']
                seq_len = self.use_seq_length if self.use_seq_length > 0 else len(
                    tokens)
                if len(tokens) <= seq_len:
                    start_idx = 0
                else:
                    start_idx = random.randint(0, len(tokens) - seq_len - 1)
                yield tokens[
                      start_idx:start_idx + seq_len]  # return list of int


def _main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train', default=False, action='store_true', help="")
    parser.add_argument('--data_path', type=str,
                        help="data path for train or inference")
    parser.add_argument('--model_path', type=str,
                        help="model path for train or inference")
    parser.add_argument('--order', type=int, default=6,
                        help="maximum of ngram length")
    parser.add_argument('--max_seq_id', type=int, default=4096,
                        help="max value of integer of sequence")
    parser.add_argument('--max_vocab_size', type=int, default=50_000,
                        help="vocab size of tokens, must be larger than max seq id")
    parser.add_argument('--min_count', type=int, default=1,
                        help="if a label`s count < min_count, it will not be saved into memory")
    parser.add_argument('--use_seq_length', type=int, default=375,
                        help="only use chunk of integer sequence to avoid memory leaking")
    parser.add_argument('--workers', type=int, default=4,
                        help="parallel workers")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="batch size for parallel model")
    args = parser.parse_args()
    if args.train:
        assert not os.path.exists(args.model_path)
        trainer = Trainer(max_seq_id=args.max_seq_id,
                          order=args.order,
                          max_vocab_size=args.max_vocab_size,
                          min_count=args.min_count,
                          )
        trainer.train(Corpus(data_path=args.data_path,
                             use_seq_length=args.use_seq_length),
                      workers=args.workers,
                      batch_size=args.batch_size)
        trainer.save(args.model_path)
        logger_x.info("Save model to {}".format(args.model_path))

        logger_x.info("analysis for ngrams")
        pieces = json.load(open(args.model_path))
        new_pieces = dict()
        for k_str, v in pieces.items():
            k = [int(x) for x in k_str.split()]
            k = tuple(k)
            new_pieces[k] = v
        pieces = new_pieces
        for i in range(args.order):
            pieces_i = {k: v for k, v in pieces.items() if len(k) == i}
            logger_x.info("ngram-{}: {}".format(i, len(pieces_i)))
    else:
        logger_x.info("Start evaluation")
        tokenizer = Tokenizer(args.model_path)
        text_num, ids_num = 0, 0
        for idx, text in enumerate(
                Corpus(data_path=args.data_path, use_seq_length=-1)):
            ids = tokenizer.encode(text)
            res = tokenizer.decode(ids)
            text_num += len(text)
            ids_num += len(ids)
            if idx % 1000 == 0:
                assert text == res
                logger_x.info(
                    "idx:{}, {} -> {}, total compress rate: {}".format(
                        idx, len(text), len(ids), ids_num / text_num))
            if idx > 10_0000:
                break
        logger_x.info("final rate: {}".format(ids_num / text_num))
    return


if __name__ == '__main__':
    _main()
    pass
