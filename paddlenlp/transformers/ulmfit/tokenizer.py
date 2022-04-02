# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter, defaultdict
import html
import numpy as np
import re

import paddle
from paddle.utils import try_import

__all__ = [
    "SpacyTokenizer"
]

# special tokens
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split(
)
default_text_spec_tok = [
    UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ
]

# Cell
_re_spec = re.compile(r'([/#\\])')


def spec_add_spaces(t):
    "Add spaces around / and #"
    return _re_spec.sub(r' \1 ', t)


# Cell
_re_space = re.compile(' {2,}')


def rm_useless_spaces(t):
    "Remove multiple spaces"
    return _re_space.sub(' ', t)


# Cell
_re_rep = re.compile(r'(\S)(\1{2,})')


def replace_rep(t):
    "Replace repetitions at the character level: cccc -- TK_REP 4 c"

    def _replace_rep(m):
        c, cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    return _re_rep.sub(_replace_rep, t)


# Cell
_re_wrep = re.compile(r'(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)')


# Cell
def replace_wrep(t):
    "Replace word repetitions: word word word word -- TK_WREP 4 word"

    def _replace_wrep(m):
        c, cc, e = m.groups()
        return f' {TK_WREP} {len(cc.split())+2} {c} {e}'

    return _re_wrep.sub(_replace_wrep, t)


# Cell
def fix_html(x):
    "Various messy things we've seen in documents"
    x = x.replace('#39;', "'").replace('amp;', '&').replace(
        '#146;', "'").replace('nbsp;', ' ').replace('#36;', '$').replace(
            '\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
                '\\"', '"').replace('<unk>', UNK).replace(' @.@ ', '.').replace(
                    ' @-@ ', '-').replace('...', ' …')
    return html.unescape(x)


# Cell
_re_all_caps = re.compile(r'(\s|^)([A-Z]+[^a-z\s]*)(?=(\s|$))')


# Cell
def replace_all_caps(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."

    def _replace_all_caps(m):
        tok = f'{TK_UP} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"

    return _re_all_caps.sub(_replace_all_caps, t)


# Cell
_re_maj = re.compile(r'(\s|^)([A-Z][^A-Z\s]*)(?=(\s|$))')


# Cell
def replace_maj(t):
    "Replace tokens in Sentence Case by their lower version and add `TK_MAJ` before."

    def _replace_maj(m):
        tok = f'{TK_MAJ} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"

    return _re_maj.sub(_replace_maj, t)


# Cell
def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else
            '') + t.lower().strip() + (f' {EOS}' if add_eos else '')


# Cell
def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '▁')


defaults_text_proc_rules = [
    fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces,
    replace_all_caps, replace_maj, lowercase
]
defaults_text_postproc_rules = [replace_space]


def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x


# Cell
def compose(*funcs):
    "Create a function that composes all functions in `funcs`, passing along remaining `*args` and `**kwargs` to all"
    funcs = list(funcs)
    if len(funcs) == 0: return noop
    if len(funcs) == 1: return funcs[0]

    def _inner(x, *args, **kwargs):
        for f in funcs:
            x = f(x, *args, **kwargs)
        return x

    return _inner


# Cell
def maps(*args):
    "Like `map`, except funcs are composed first"
    f = compose(*args[:-1])

    def _f(b):
        return f(b)

    return map(_f, args[-1])


# Cell
def save_encoder(model, encoder_path):
    encoder_dict = {}
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if "encoder" in key:
            new_key = key.replace(".encoder", ".module.encoder")
            encoder_dict[new_key] = state_dict[key]
    paddle.save(encoder_dict, encoder_path)


# Cell
def load_encoder(model, encoder_path):
    encoder_dict = paddle.load(encoder_path)
    state_dict = model.state_dict()
    new_state_dict = {}
    for key in encoder_dict.keys():
        new_state_dict[key] = encoder_dict[key]

    for key in state_dict.keys():
        if "encoder" not in key:
            new_state_dict[key] = state_dict[key]

    return model


class SpacyTokenizer():
    "Spacy tokenizer for `lang`"

    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        if special_toks is None:
            self.special_toks = default_text_spec_tok

        nlp = spacy.blank(lang)
        for w in self.special_toks:
            nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe, self.buf_sz = nlp.pipe, buf_sz

    def __call__(self, items):
        return (list(map(str, list(doc)))
                for doc in self.pipe(map(str, items), batch_size=self.buf_sz))


#LMTokenizer如何和CLSTokenizer合并？
#CLSTokenizer如何和CLSTokenizer合并？
class LMTokenizer:
    min_freq = 3
    max_vocab = 60000
    special_toks = default_text_spec_tok
    vocab = None
    vocab_size = None
    o2i = None
    spacy_tok = None
    full_tokens = []
    full_lens = []
    cumlens = 0
    totlen = 0

    def __init__(self, text, bs=2, seq_len=72, backwards=False):
        self.backwards = backwards
        self.bs = bs
        self.seq_len = seq_len
        self.spacy_tok = SpacyTokenizer()
        tok_res = self.get_spacy_toks(text)
        char_tokens = []
        self.full_tokens = []
        for tr in tok_res:
            char_tokens += tr
            self.full_lens.append(len(tr))
            self.full_tokens.append(tr)

        corpus = ((sum(self.full_lens) - 1) / bs) * bs
        self.bl = corpus // bs  # bl stands for batch length
        self.n_batches = self.bl // (seq_len) + int(self.bl % seq_len != 0)
        self.n = int((self.n_batches - 1) * bs)
        self.cumlens = np.cumsum(([0] + self.full_lens))
        self.totlen = self.cumlens[-1]
        self.last_len = self.bl - (self.n_batches - 1) * seq_len

        count = Counter(char_tokens)
        self.vocab = self.make_vocab(count)
        self.vocab_size = len(self.vocab)
        self.o2i = defaultdict(int, {v: k for k, v in enumerate(self.vocab) if v != 'xxfake'})
        self.full_tokens = self.convert_token_id(self.full_tokens)

    def get_spacy_toks(self, text):
        tok_res = self.spacy_tok(maps(*defaults_text_proc_rules, text))
        return (list(maps(*defaults_text_postproc_rules, o)) for o in tok_res)

    def make_vocab(self, count):
        vocab = [o for o, c in count.most_common(self.max_vocab) if c >= self.min_freq]
        for o in reversed(self.special_toks):  # Make sure all special tokens are in the vocab
            if o in vocab: vocab.remove(o)
            vocab.insert(0, o)
        vocab = vocab[:self.max_vocab]
        vocab = vocab + [f'xxfake' for i in range(0, 8 - len(vocab) % 8)]
        return vocab

    def __call__(self, o):
        words = self.get_spacy_toks(o)
        return self.convert_token_id(words)

    def convert_token_id(self, words):
        all = []
        for o_ in list(words):
            tmp = []
            for x in o_:
                tmp.append(self.o2i[x])
            if self.backwards:
                tmp.reverse()
            all.append(tmp)
        return all

    def decode(self, o):
        return [self.vocab[o_] for o_ in o]

    def create_idxs(self):
        return [i for i in range(self.n)]

    def doc_idx(self, i):
        if i < 0: i = self.totlen + i
        docidx = np.searchsorted(self.cumlens, i + 1) - 1
        cl = self.cumlens[docidx]
        return docidx, i - cl

    def get_item(self, seq):
        if seq is None: seq = 0
        if seq >= self.n: raise IndexError
        i_stop = self.last_len if seq // self.bs == self.n_batches - 1 else self.seq_len
        i_start = int((seq % self.bs) * self.bl + (seq // self.bs) * self.seq_len)
        i_stop = int(i_stop + i_start + 1)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.totlen + 1

        st_d, st_i = self.doc_idx(i_start)
        en_d, en_i = self.doc_idx(i_stop)
        res = self.full_tokens[st_d][st_i:(en_i if st_d == en_d else sys.maxsize)]
        for b in range(st_d + 1, en_d): res += self.full_tokens[b]
        if st_d != en_d and en_d < len(self.full_tokens): res += self.full_tokens[en_d][:en_i]

        if len(res) != 73:
            return None, None
        return res[:-1], res[1:]



