from __future__ import annotations
from enum import Enum
import torch
import itertools


class PairEnum(list):
    """
    Store a set of pairs of atom enums, as well as their integer
    representation in a tensor.
    """

    def __init__(
        self: PairEnum,
        bonds: list[tuple[Enum, Enum]]
    ) -> None:

        super().__init__(bonds)

    def __add__(
        self: PairEnum,
        other: list,
    ) -> PairEnum:

        return self.__class__(super().__add__(other))

    def indices(
        self: PairEnum,
    ) -> torch.Tensor:

        return  torch.tensor(
            [[atom1.value, atom2.value]
             for atom1, atom2 in self]
        )

    def pairwise(
        self: PairEnum,
    ) -> torch.Tensor:

        n = self.indices().max() + 1
        table = torch.ones(n, n, dtype=torch.long) * -1

        ix = 0
        for x, y in self:
            table[x.value, y.value] = ix
            table[y.value, x.value] = ix
            ix += 1

        return table


class IndexEnum(Enum):
    """
    An enum with the ability to collect all enum values into a tensor.
    """

    @classmethod
    def index(
        cls: type[IndexEnum],
    ) -> torch.Tensor:
        """
        Return a tensor of all the indices in the enum.
        """

        return torch.Tensor([
            atom.value for atom in cls
        ]).long()

    @classmethod
    def list(
        cls: type[IndexEnum],
        modifier: str = ''
    ) -> list[str]:
        """
        Return the names in the enum as a list.
        """

        return [
            modifier + field.name
            for field in cls
        ]

    @classmethod
    def dict(
        cls: type[IndexEnum],
        modifier: str = ''
    ) -> dict[str, int]:
        """
        Return the enum as a dict.
        """

        return {
            modifier + field.name: field.value
            for field in cls
        }

    @classmethod
    def revdict(
        cls: type[IndexEnum],
        modifier: str = ''
    ) -> dict[int, str]:
        """
        Return the enum as a reversed dict.
        """

        return {
            field.value: modifier + field.name
            for field in cls
        }

    @classmethod
    def pairs(
        cls: type[IndexEnum],
    ) -> PairEnum:
        """
        Return all pairs of objects, without repeats.
        """

        pairs = []
        for x, y in itertools.product(cls, cls):
            if (x, y) not in pairs and (y, x) not in pairs:
                pairs.append((x, y))

        return PairEnum(pairs)


class Residue(IndexEnum):
    A   = 0
    DA  = 0
    C   = 1
    DC  = 1
    G   = 2
    DG  = 2
    U   = 3
    DU  = 3
    T   = 4
    DT  = 4
    ALA = 5
    CYS = 6
    ASP = 7
    GLU = 8
    PHE = 9
    GLY = 10
    HIS = 11
    ILE = 12
    LYS = 13
    LEU = 14
    MET = 15
    ASN = 16
    PRO = 17
    GLN = 18
    ARG = 19
    SER = 20
    THR = 21
    VAL = 22
    TRP = 23
    TYR = 24

RES_ABBREV = {
    'A': 'a',
    'C': 'c',
    'G': 'g',
    'U': 'u',
    'T': 't',
    'N' : 'n',
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y'
}


class Adenosine(IndexEnum):
    OP3  = 1
    P    = 2
    OP1  = 3
    OP2  = 4
    O5p  = 5
    C5p  = 6
    C4p  = 7
    O4p  = 8
    C3p  = 9
    O3p  = 10
    C2p  = 11
    O2p  = 12
    C1p  = 13
    N9   = 14
    C8   = 15
    N7   = 16
    C5   = 17
    C6   = 18
    N6   = 19
    N1   = 20
    C2   = 21
    N3   = 22
    C4   = 23
    HOP3 = 24
    HOP2 = 25
    H5p  = 26
    H5pp = 27
    H4p  = 28
    H3p  = 29
    HO3p  = 30
    H2p  = 31
    HO2p  = 32
    HO5p = 148
    H1p  = 33
    H8   = 34
    H61  = 35
    H62  = 36
    H2   = 37

class Cytosine(IndexEnum):
    OP3  = 38
    P    = 39
    OP1  = 40
    OP2  = 41
    O5p  = 42
    C5p  = 43
    C4p  = 44
    O4p  = 45
    C3p  = 46
    O3p  = 47
    C2p  = 48
    O2p  = 49
    C1p  = 50
    N1   = 51
    C2   = 52
    O2   = 53
    N3   = 54
    C4   = 55
    N4   = 56
    C5   = 57
    C6   = 58
    HOP3 = 59
    HOP2 = 60
    H5p  = 61
    H5pp = 62
    H4p  = 63
    H3p  = 64
    HO3p  = 65
    H2p  = 66
    HO2p  = 67
    HO5p = 145
    H1p  = 68
    H41  = 69
    H42  = 70
    H5   = 71
    H6   = 72

class Guanosine(IndexEnum):
    OP3  = 73
    P    = 74
    OP1  = 75
    OP2  = 76
    O5p  = 77
    C5p  = 78
    C4p  = 79
    O4p  = 80
    C3p  = 81
    O3p  = 82
    C2p  = 83
    O2p  = 84
    C1p  = 85
    N9   = 86
    C8   = 87
    N7   = 88
    C5   = 89
    C6   = 90
    O6   = 91
    N1   = 92
    C2   = 93
    N2   = 94
    N3   = 95
    C4   = 96
    HOP3 = 97
    HOP2 = 98
    H5pp = 99
    H5p  = 100
    H4p  = 101
    H3p  = 102
    HO3p  = 103
    H2p  = 104
    HO2p  = 105
    HO5p = 146
    H1p  = 106
    H8   = 107
    H1   = 108
    H21  = 109
    H22  = 110

class Uridine(IndexEnum):
    OP3  = 111
    P    = 112
    OP1  = 113
    OP2  = 114
    O5p  = 115
    C5p  = 116
    C4p  = 117
    O4p  = 118
    C3p  = 119
    O3p  = 120
    C2p  = 121
    O2p  = 122
    C1p  = 123
    N1   = 124
    C2   = 125
    O2   = 126
    N3   = 127
    C4   = 128
    O4   = 129
    C5   = 130
    C6   = 131
    HOP3 = 132
    HOP2 = 133
    H5p  = 134
    H5pp = 135
    H4p  = 136
    H3p  = 137
    HO3p  = 138
    H2p  = 139
    HO2p  = 140
    HO5p = 147
    H1p  = 141
    H3   = 142
    H5   = 143
    H6   = 144

RibonucleicAcid = IndexEnum("RibonucleicAcid", Adenosine.dict() | Cytosine.dict() | Guanosine.dict() | Uridine.dict())

_A = {"A_" + key: value for key, value in Adenosine.dict().items() if 'p' in key or "P" in key}
_C = {"C_" + key: value for key, value in Cytosine.dict().items() if 'p' in key or "P" in key}
_G = {"G_" + key: value for key, value in Guanosine.dict().items() if 'p' in key or "P" in key}
_U = {"U_" + key: value for key, value in Uridine.dict().items() if 'p' in key or "P" in key}

Backbone = IndexEnum("Backbone", _A | _C | _G | _U)

_A = {"A_" + key: value for key, value in Adenosine.dict().items() if 'p' not in key and "P" not in key}
_C = {"C_" + key: value for key, value in Cytosine.dict().items() if 'p' not in key and "P" not in key}
_G = {"G_" + key: value for key, value in Guanosine.dict().items() if 'p' not in key and "P" not in key}
_U = {"U_" + key: value for key, value in Uridine.dict().items() if 'p' not in key and "P" not in key}

Nucleobase = IndexEnum("Nucleobase", _A | _C | _G | _U)

_A = {"A_" + key: value for key, value in Adenosine.dict().items() if "P" in key}
_C = {"C_" + key: value for key, value in Cytosine.dict().items() if "P" in key}
_G = {"G_" + key: value for key, value in Guanosine.dict().items() if "P" in key}
_U = {"U_" + key: value for key, value in Uridine.dict().items() if "P" in key}

Phosphate = IndexEnum("Phosphate", _A | _C | _G | _U)



COARSE = torch.tensor([
    # Adenosine.C1p.value,
    # Cytosine.C1p.value,
    # Guanosine.C1p.value,
    # Uridine.C1p.value,
    Adenosine.N1.value,
    Cytosine.N3.value,
    Guanosine.N1.value,
    Uridine.N3.value,
])



class Element(IndexEnum):
    H = 1
    C = 6
    N = 7
    O = 8
    P = 15
    S = 16
