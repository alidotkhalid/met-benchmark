"""Semantic-equivalence guard for adaptive-attack construction.

The benchmark scores with containment match, so a closed-book answer can be marked 'wrong' while
being the same answer in another surface form ('Three' vs '3', 'the USA' vs 'United States').
Adopting such an answer as the attack target would produce an 'attack' whose target is the truth.
These checks are used only to filter attack targets; they never touch scoring.
"""
import re

_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
        "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
        "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
        "hundred": "100", "thousand": "1000", "million": "1000000", "billion": "1000000000",
        "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
        "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}
_DROP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "are", "was", "were",
         "st", "st.", "saint", "mr", "mrs", "dr", "approximately", "about", "around", "roughly"}


def canon(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    w = [_NUM.get(x, x) for x in s.split() if x not in _DROP]
    return " ".join(w)


def equivalent(a, b):
    """True if a and b are plausibly the same answer under surface variation."""
    ca, cb = canon(a), canon(b)
    if not ca or not cb:
        return False
    if ca == cb or ca.replace(" ", "") == cb.replace(" ", ""):
        return True
    sa, sb = set(ca.split()), set(cb.split())
    # one answer's tokens wholly contained in the other's ('4 years' vs 'typically 4 years').
    # Partial overlap is NOT equivalence: 'World War II'/'World War I' and
    # 'March 31, 2016'/'March 31, 2001' share most tokens and are different answers.
    return sa <= sb or sb <= sa
