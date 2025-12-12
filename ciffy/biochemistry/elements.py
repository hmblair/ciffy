"""
Chemical element definitions.
"""

from ..utils import IndexEnum


class Element(IndexEnum):
    """
    Chemical elements with their atomic numbers.

    Values correspond to atomic numbers for common biological elements.
    """

    # Common organic elements
    H = 1
    C = 6
    N = 7
    O = 8
    P = 15
    S = 16

    # Ion elements (for ION_COMP_IDS recognition)
    LI = 3    # Lithium
    F = 9     # Fluorine
    NA = 11   # Sodium
    MG = 12   # Magnesium
    AL = 13   # Aluminum
    CL = 17   # Chlorine
    K = 19    # Potassium
    CA = 20   # Calcium
    MN = 25   # Manganese
    FE = 26   # Iron
    CO = 27   # Cobalt
    NI = 28   # Nickel
    CU = 29   # Copper
    ZN = 30   # Zinc
    BR = 35   # Bromine
    RB = 37   # Rubidium
    SR = 38   # Strontium
    CD = 48   # Cadmium
    I = 53    # Iodine
    CS = 55   # Cesium
    BA = 56   # Barium
    HG = 80   # Mercury
    PB = 82   # Lead
