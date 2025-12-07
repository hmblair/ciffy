"""
Chemical element definitions.
"""

from ..utils import IndexEnum


class Element(IndexEnum):
    """
    Chemical elements with their atomic numbers.

    Values correspond to atomic numbers for common biological elements.
    """

    H = 1
    C = 6
    N = 7
    O = 8
    P = 15
    S = 16
