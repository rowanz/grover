"""Byte pair encoding utilities

Some functions are adapted from OpenAI but with modifications

https://github.com/openai/gpt-2
"""

import os
import json
import regex as re
from functools import lru_cache
import tensorflow as tf
import random
import numpy as np


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = {k: v + 1 for k, v in encoder.items()}
        self.encoder['<|padding|>'] = 0
        self.padding = 0

        del self.encoder['<|endoftext|>']

        for special_token_type in ['domain', 'date', 'authors', 'title', 'article', 'summary']:
            setattr(self, f'begin_{special_token_type}', len(self.encoder))
            self.encoder[f'<|begin{special_token_type}|>'] = len(self.encoder)

            setattr(self, f'end_{special_token_type}', len(self.encoder))
            self.encoder[f'<|endof{special_token_type}|>'] = len(self.encoder)

        # This will be used if we want to combine short articles.
        self.reset_context = len(self.encoder)
        self.encoder['<|resetcontext|>'] = len(self.encoder)

        ################################## END OF SPECIAL TOKENS TO ADD

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def __len__(self):
        return len(self.encoder)

    @property
    def special_tokens_onehot(self):
        """ Return the IDs of all special tokens"""
        return [(self.decoder[i].startswith('<|') and self.decoder[i].endswith('|>')) for i in range(len(self))]


def get_encoder():
    directory_name = os.path.dirname(__file__)
    with open(os.path.join(directory_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(directory_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


##############################################################
# TURN SOMETHING INTO THE RIGHT FORMAT FOR AN EXAMPLE
##############################################################
def _tokenize_article_pieces(encoder, item):
    """
    Turn the article into tokens
    NOTE: in hindsight I kinda messed up here because the first token is always represented as a BPE continuation
    rather than an initial token in its own right. whoops....

    :param item: Contains things that need to be tokenized


    fields are ['domain', 'date', 'authors', 'title', 'article', 'summary']
    :return: dict
    """
    article_pieces = {
        'article': [encoder.begin_article] + encoder.encode(item['text']) + [encoder.end_article],
        'domain': [encoder.begin_domain] + encoder.encode(item['domain']) + [encoder.end_domain],
        'title': [encoder.begin_title] + encoder.encode(item['title']) + [encoder.end_title],
    }
    # 4/6: Attach the summary too, why the hell not
    if item['summary'] and len(item['summary']) > 50:
        article_pieces['summary'] = [encoder.begin_summary] + encoder.encode(item['summary']) + [encoder.end_summary]

    # 5/6: date
    date_split = item['publish_date'].split('-')
    assert len(date_split) == 3
    assert date_split[0].isdigit()

    date_txt = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December'][int(date_split[0]) - 1] + ' {}, {}'.format(
        date_split[1], date_split[2])
    article_pieces['date'] = [encoder.begin_date] + encoder.encode(date_txt) + [encoder.end_date]

    # 6/6: authors
    authors = ', '.join(item['authors'])
    if len(authors) > 5:
        article_pieces['authors'] = [encoder.begin_authors] + encoder.encode(authors) + [encoder.end_authors]
    return article_pieces


def _cut_tokens_to_add_stuff(tokens, stuff_to_add, desired_size, padding_token):
    """
    The idea behind this function is to take away tokens from `tokens' such that tokens[:LENGTH] + stuff_to_add becomes
    exactly at the right size (desired_size).

    :param tokens:
    :param stuff_to_add:
    :param desired_size:
    :return:
    """
    if len(tokens) >= desired_size:
        return tokens

    # no way we can add this stuff
    if len(stuff_to_add) >= desired_size:
        return tokens

    if (len(tokens) + len(stuff_to_add)) <= desired_size:
        return tokens + stuff_to_add

    # Otherwise we'll have to actually cut
    tokens = tokens[:(desired_size - len(stuff_to_add) - 1)]
    tokens.append(padding_token)
    tokens.extend(stuff_to_add)
    return tokens


def tokenize_for_grover_training(encoder, item, desired_size=1024, unconditional_prob=0.35, metadata_dropout_prob=0.1,
                                 cut_prob=0.2):
    """
    Not only will we tokenize an item with a BPE encoder, but we'll also put it in a nice format for language modeling.
    The goal is to MINIMIZE PADDING. If we don't fill up the desired size of 1024 tokens then we're wasting compute.

    The canonical order is

    DOMAIN DATE AUTHORS TITLE ARTICLE SUMMARY


    :param encoder:
    :param item: Contains things like
          {"url": "https://www.advocate.com/node/1010911",
          "timestamp": "20180118211607",
           "url_used": "https://web.archive.org/web/20180118211607id_/https://www.advocate.com/node/1010911",
           "domain": "advocate.com",
           "title": "Report: One-Third of Trump's Judicial Picks Are Anti-LGBT",
           "text": ....
           "summary": ....
           "authors": list
           "publish_date": ...
           }
    :param desired_size: the goal for how long the span will be
    :param unconditional_prob: The probability that we will generate JUST THE TEXT first.
    :param metadata_dropout_prob: The probability that we will drop out each item of metadata
    :param cut_prob: The probability that, if we're already over the desired size, we'll cut the article and start
                    predicting metadata before the desired_size window ends.
    :return:
    """
    # Get all the bits and pieces
    article_pieces = _tokenize_article_pieces(encoder, item)
    canonical_metadata_order = ['domain', 'date', 'authors', 'title']

    # unconditional_prob is probability we only generate the text first, without any metadata
    switch = random.random()
    if switch < unconditional_prob:
        assignments = {'article': 'a'}
        chunk_a = article_pieces.pop('article')
        chunk_b = []
        for x in canonical_metadata_order + ['summary']:
            if random.random() > metadata_dropout_prob:
                chunk_b.extend(article_pieces.pop(x, []))
                assignments[x] = 'b'
    elif switch < 0.5:
        # Put everything in chunk_a, without dropout
        assignments = {}
        chunk_a = []
        chunk_b = []
        for x in canonical_metadata_order + ['article', 'summary']:
            chunk_a.extend(article_pieces.pop(x, []))
            assignments[x] = 'a'
    else:
        assignments = {}
        chunk_a = []
        chunk_b = []
        for k in canonical_metadata_order + ['article', 'summary']:
            if random.random() < metadata_dropout_prob and k not in ('article', 'title'):
                pass
            elif random.random() < 0.5:
                if k != 'summary':
                    chunk_a.extend(article_pieces.pop(k, []))
                    assignments[k] = 'a'
            else:
                chunk_b.extend(article_pieces.pop(k, []))
                assignments[k] = 'b'

    if (len(chunk_a) + len(chunk_b)) <= desired_size:
        return chunk_a + chunk_b

    if (assignments.get('article', '') == 'a') and (len(chunk_b) > 0) and (random.random() < cut_prob):
        return _cut_tokens_to_add_stuff(chunk_a, chunk_b, desired_size, encoder.padding)

    tokens = chunk_a + chunk_b
    return tokens


def detokenize(encoder, tokens):
    return encoder.decode(tokens)


#######################################

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def sliding_window(article, max_seq_length, pad_token):
    """
    Randomly sample some spans. It's a simple approximation of sliding window
    :param tokens:
    :param max_seq_length:
    :return:
    """
    # if it's shorter, no need for this
    if len(article['input_ids']) <= max_seq_length:
        amount_to_pad = max_seq_length - len(article['input_ids'])
        article['input_ids'].extend([pad_token] * amount_to_pad)
        yield article
        return

    num_spans = len(article['input_ids']) - max_seq_length + 1
    weights = np.ones(num_spans, dtype=np.float32)
    # weights[0] = max_seq_length
    weights /= weights.sum()

    num_to_yield = int(0.5 + len(article['input_ids']) / max_seq_length)
    starts = np.random.choice(num_spans, size=num_to_yield, replace=False, p=weights)

    input_ids = article.pop('input_ids')
    for i in starts.tolist():
        article['input_ids'] = input_ids[i:(i + max_seq_length)]
        yield article


def format_context(encoder, news_article, target):
    """
    Generates a news article given some partial information
    :param news_article: Contains context
    :param target: What we want to get an answer for.
    :return:
    """
    canonical_metadata_order = ['domain', 'date', 'authors', 'title', 'article']
    tokens = []
    for metadata_category in canonical_metadata_order:
        metadata = news_article.get(metadata_category, '').strip()

        # This MIGHT BE needed because I think during training time we never saw empty articles
        # if metadata or ((metadata_category == 'article') and target != 'article'):
        if (metadata_category == 'article') and (target != 'article'):
            metadata = news_article.get('title', '')  # Just copy from the title maybe?

        if metadata:
            tokens.append(encoder.__dict__[f'begin_{metadata_category}'])
            tokens.extend(encoder.encode(metadata))
            tokens.append(encoder.__dict__[f'end_{metadata_category}'])

    assert target in (canonical_metadata_order + ['summary'])
    tokens.append(encoder.__dict__[f'begin_{target}'])
    return tokens


def extract_generated_target(output_tokens, encoder, target):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_tokens = output_tokens == encoder.__dict__[f'begin_{target}']
    if np.any(start_tokens):
        start_ind = np.argmax(start_tokens) + 1
    else:
        start_ind = 0

    end_tokens = output_tokens == encoder.__dict__[f'end_{target}']
    if np.any(end_tokens):
        end_ind = np.argmax(end_tokens)
    else:
        end_ind = output_tokens.shape[0]

    return {
        'extraction': encoder.decode(output_tokens[start_ind:end_ind]),
        'start_ind': start_ind,
        'end_ind': end_ind,
    }


if __name__ == '__main__':
    encoder = get_encoder()
    print("VOCAB SIZE IS {}".format(len(encoder.encoder)))
