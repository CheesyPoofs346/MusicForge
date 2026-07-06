"""Phoneme conversion matching Muse's training format, e.g.
    [lyrics:\n<line>\n<line>][phoneme:\n<sm> <ym> sp ...\n...]
Ported from pro/Muse/data_pipeline/meta_process/meta_phonemes.py (Muse's own
training-data builder) with the my_tool.py / poly_correct.json / lexicon.txt
dependencies dropped — those files aren't shipped in the cloned repo, and
they're refinements (heteronym correction, a manual English-word override
dict) rather than something the format needs to be valid. g2p_en covers
every English word directly.
"""
import re
import string

import jieba
from g2p_en import G2p
from pypinyin import Style, pinyin
from pypinyin_dict.phrase_pinyin_data import cc_cedict

cc_cedict.load()
_g2p = G2p()

_RE_SPECIAL_PINYIN = re.compile(r'^(n|ng|m)$')
_CH_PUNCT = r'[。，？！；：“”‘’《》〈〉【】『』—…、（）]'


def _split_py(py: str) -> tuple[str, str]:
    tone = py[-1]
    py = py[:-1]
    sm = ""
    suf_r = ""
    if _RE_SPECIAL_PINYIN.match(py):
        py = 'e' + py
    if py and py[-1] == 'r':
        suf_r = 'r'
        py = py[:-1]
    if not py:
        return "", suf_r + tone
    if py in ('zi', 'ci', 'si', 'ri'):
        sm, ym = py[:1], "ii"
    elif py in ('zhi', 'chi', 'shi'):
        sm, ym = py[:2], "iii"
    elif py in ('ya', 'yan', 'yang', 'yao', 'ye', 'yong', 'you'):
        sm, ym = "", 'i' + py[1:]
    elif py in ('yi', 'yin', 'ying'):
        sm, ym = "", py[1:]
    elif py in ('yu', 'yv', 'yuan', 'yvan', 'yve', 'yun', 'yvn'):
        sm, ym = "", 'v' + py[2:]
    elif py == 'wu':
        sm, ym = "", "u"
    elif py[0] == 'w':
        sm, ym = "", "u" + py[1:]
    elif len(py) >= 2 and py[0] in ('j', 'q', 'x') and py[1] == 'u':
        sm, ym = py[0], 'v' + py[2:]
    else:
        seg = re.search('a|e|i|o|u|v', py)
        if not seg:
            return "", ""
        sm, ym = py[:seg.start()], py[seg.start():]
        ym = {'ui': 'uei', 'iu': 'iou', 'un': 'uen', 'ue': 've'}.get(ym, ym)
    return sm, ym + suf_r + tone


def _trans_cn(text: str) -> list[str]:
    phonemes = []
    for seg in jieba.cut(text):
        if not seg.strip():
            continue
        pys = [p[0] for p in pinyin(seg, style=Style.TONE3, neutral_tone_with_five=True)]
        if any(re.search(_CH_PUNCT, p) or p in string.punctuation for p in pys):
            continue
        for py in pys:
            sm, ym = _split_py(py)
            if sm:
                phonemes.append(sm)
            if ym:
                phonemes.append(ym)
        phonemes.append("sp")
    return phonemes


def _trans_en(word: str) -> list[str]:
    phonemes = [p for p in _g2p(word.lower()) if p != " "]
    if phonemes:
        phonemes.append("sp")
    return phonemes


def _char_lang(c: str) -> int:
    if '一' <= c <= '鿿':
        return 0
    if ('a' <= c <= 'z') or ('A' <= c <= 'Z'):
        return 1
    return 3


def _lang_segments(text: str) -> list[tuple[str, int]]:
    segs, tag, buf = [], -1, ""
    for c in text:
        lang = _char_lang(c)
        if lang != tag:
            if buf:
                segs.append((buf, tag))
            buf = ""
            tag = lang
        if lang < 2:
            buf += c
    if buf:
        segs.append((buf, tag))
    return segs


def phonemes_for_line(line: str) -> str:
    """Space-joined phonemes (with 'sp' word breaks) for one lyric line."""
    phonemes = []
    for seg, tag in _lang_segments(line):
        phonemes += _trans_cn(seg) if tag == 0 else _trans_en(seg)
    return " ".join(phonemes)


def phoneme_block(lyric_lines: list[str]) -> str:
    """Build the '[phoneme:\\n...\\n...]' block matching one lyric line per row,
    tone digits stripped (Muse's training data strips them too)."""
    block = "[phoneme:" + "\n".join(phonemes_for_line(line) for line in lyric_lines) + "]"
    return re.sub(r'\d+', '', block)
