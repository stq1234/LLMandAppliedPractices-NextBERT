"""SQuAD 1.1 and CMRC2018 text metrics, with offline NLTK assets.

CMRC reference: ymcui/cmrc2018 baseline/cmrc2018_evaluate.py (v6),
https://github.com/ymcui/cmrc2018, CC-BY-SA-4.0. Its LCS is a longest
CONTIGUOUS common token substring. English segmentation uses NLTK, not jieba.
"""
from collections import Counter
from pathlib import Path
import re
import string

CMRC_PUNCTUATION = set([
    '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
    '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……',
    '·', '、', '「', '」', '（', '）', '－', '～', '『', '』',
])


def squad_normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def cmrc_normalize(text):
    return ''.join(c for c in str(text).lower().strip() if c not in CMRC_PUNCTUATION)


def cmrc_tokens(text):
    import nltk
    assets = Path(__file__).resolve().parents[1] / 'artifacts' / 'nltk_data'
    if str(assets) not in nltk.data.path:
        nltk.data.path.insert(0, str(assets))
    result, buffer = [], []

    def flush():
        if buffer:
            result.extend(nltk.word_tokenize(''.join(buffer)))
            buffer.clear()

    for char in str(text).lower().strip():
        if char in CMRC_PUNCTUATION:
            continue
        if '\u4e00' <= char <= '\u9fa5':
            flush()
            result.append(char)
        else:
            buffer.append(char)
    flush()
    return result


def longest_common_substring(a, b):
    previous = [0] * (len(b) + 1)
    longest = 0
    for left in a:
        current = [0] * (len(b) + 1)
        for j, right in enumerate(b, 1):
            if left == right:
                current[j] = previous[j - 1] + 1
                longest = max(longest, current[j])
        previous = current
    return longest


def answer_f1(prediction, gold, language):
    if language == 'zh':
        p, g = cmrc_tokens(prediction), cmrc_tokens(gold)
        overlap = longest_common_substring(p, g)
    else:
        p, g = squad_normalize(prediction).split(), squad_normalize(gold).split()
        if not p and not g:
            return 1.0
        overlap = sum((Counter(p) & Counter(g)).values())
    return 2.0 * overlap / (len(p) + len(g)) if overlap else 0.0


def answer_em(prediction, gold, language):
    normalize = cmrc_normalize if language == 'zh' else squad_normalize
    return float(normalize(prediction) == normalize(gold))
