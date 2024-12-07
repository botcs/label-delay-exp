"""
List all combinations with repetition of length n from categories ["N", "W", "R"].
"""

import itertools


def combinations_with_replacement(n, categories):
    """
    List all combinations with repetition of length n from categories ["N", "W", "R"].
    """
    return list(itertools.combinations_with_replacement(categories, n))

for c in combinations_with_replacement(4, ["N", "W", "R"]):
    print(f"- {''.join(c)}")