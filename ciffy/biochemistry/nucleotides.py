"""
Nucleotide atom definitions.

Each nucleotide class defines atoms using unique integer indices. Atoms are
grouped by their structural role:
- Phosphate group: P, OP1, OP2, OP3
- Sugar backbone: C1'-C5', O2'-O5' (denoted with 'p' for prime)
- Nucleobase: specific to each nucleotide
- Hydrogens: various H atoms
"""

from ..utils import IndexEnum


class Adenosine(IndexEnum):
    """
    Adenosine (A) atom indices.

    Adenosine is a purine nucleoside with a 9-membered nucleobase ring.
    """

    # Phosphate group
    OP3 = 1
    P = 2
    OP1 = 3
    OP2 = 4

    # Sugar backbone
    O5p = 5
    C5p = 6
    C4p = 7
    O4p = 8
    C3p = 9
    O3p = 10
    C2p = 11
    O2p = 12
    C1p = 13

    # Nucleobase
    N9 = 14
    C8 = 15
    N7 = 16
    C5 = 17
    C6 = 18
    N6 = 19
    N1 = 20
    C2 = 21
    N3 = 22
    C4 = 23

    # Hydrogens
    HOP3 = 24
    HOP2 = 25
    H5p = 26
    H5pp = 27
    H4p = 28
    H3p = 29
    HO3p = 30
    H2p = 31
    HO2p = 32
    HO5p = 148
    H1p = 33
    H8 = 34
    H61 = 35
    H62 = 36
    H2 = 37


class Cytosine(IndexEnum):
    """
    Cytosine (C) atom indices.

    Cytosine is a pyrimidine nucleoside with a 6-membered nucleobase ring.
    """

    # Phosphate group
    OP3 = 38
    P = 39
    OP1 = 40
    OP2 = 41

    # Sugar backbone
    O5p = 42
    C5p = 43
    C4p = 44
    O4p = 45
    C3p = 46
    O3p = 47
    C2p = 48
    O2p = 49
    C1p = 50

    # Nucleobase
    N1 = 51
    C2 = 52
    O2 = 53
    N3 = 54
    C4 = 55
    N4 = 56
    C5 = 57
    C6 = 58

    # Hydrogens
    HOP3 = 59
    HOP2 = 60
    H5p = 61
    H5pp = 62
    H4p = 63
    H3p = 64
    HO3p = 65
    H2p = 66
    HO2p = 67
    HO5p = 145
    H1p = 68
    H41 = 69
    H42 = 70
    H5 = 71
    H6 = 72


class Guanosine(IndexEnum):
    """
    Guanosine (G) atom indices.

    Guanosine is a purine nucleoside with a 9-membered nucleobase ring.
    """

    # Phosphate group
    OP3 = 73
    P = 74
    OP1 = 75
    OP2 = 76

    # Sugar backbone
    O5p = 77
    C5p = 78
    C4p = 79
    O4p = 80
    C3p = 81
    O3p = 82
    C2p = 83
    O2p = 84
    C1p = 85

    # Nucleobase
    N9 = 86
    C8 = 87
    N7 = 88
    C5 = 89
    C6 = 90
    O6 = 91
    N1 = 92
    C2 = 93
    N2 = 94
    N3 = 95
    C4 = 96

    # Hydrogens
    HOP3 = 97
    HOP2 = 98
    H5pp = 99
    H5p = 100
    H4p = 101
    H3p = 102
    HO3p = 103
    H2p = 104
    HO2p = 105
    HO5p = 146
    H1p = 106
    H8 = 107
    H1 = 108
    H21 = 109
    H22 = 110


class Uridine(IndexEnum):
    """
    Uridine (U) atom indices.

    Uridine is a pyrimidine nucleoside with a 6-membered nucleobase ring.
    """

    # Phosphate group
    OP3 = 111
    P = 112
    OP1 = 113
    OP2 = 114

    # Sugar backbone
    O5p = 115
    C5p = 116
    C4p = 117
    O4p = 118
    C3p = 119
    O3p = 120
    C2p = 121
    O2p = 122
    C1p = 123

    # Nucleobase
    N1 = 124
    C2 = 125
    O2 = 126
    N3 = 127
    C4 = 128
    O4 = 129
    C5 = 130
    C6 = 131

    # Hydrogens
    HOP3 = 132
    HOP2 = 133
    H5p = 134
    H5pp = 135
    H4p = 136
    H3p = 137
    HO3p = 138
    H2p = 139
    HO2p = 140
    HO5p = 147
    H1p = 141
    H3 = 142
    H5 = 143
    H6 = 144


class GuanosineTriphosphate(IndexEnum):
    """
    Guanosine-5'-triphosphate (GTP) atom indices.

    GTP is a modified nucleotide with a triphosphate group.
    Indices start at 149.
    """

    # Gamma phosphate group
    PG = 149
    O1G = 150
    O2G = 151
    O3G = 152

    # Beta phosphate group
    O3B = 153
    PB = 154
    O1B = 155
    O2B = 156

    # Alpha phosphate group
    O3A = 157
    PA = 158
    O1A = 159
    O2A = 160

    # Sugar backbone
    O5p = 161
    C5p = 162
    C4p = 163
    O4p = 164
    C3p = 165
    O3p = 166
    C2p = 167
    O2p = 168
    C1p = 169

    # Nucleobase (guanine)
    N9 = 170
    C8 = 171
    N7 = 172
    C5 = 173
    C6 = 174
    O6 = 175
    N1 = 176
    C2 = 177
    N2 = 178
    N3 = 179
    C4 = 180

    # Hydrogens
    HOG2 = 181
    HOG3 = 182
    HOB2 = 183
    H5p = 184
    H5pp = 185
    H4p = 186
    H3p = 187
    HO3p = 188
    H2p = 189
    HO2p = 190
    H1p = 191
    H8 = 192
    H1 = 193
    H21 = 194
    H22 = 195


class CytidineTriphosphate(IndexEnum):
    """
    Cytidine-5'-triphosphate (CCC) atom indices.

    CCC is a modified nucleotide found at 3' terminal.
    Indices start at 196.
    """

    # Gamma phosphate group
    PC = 196
    O1C = 197
    O2C = 198

    # Standard phosphate
    P = 199
    OP1 = 200
    OP2 = 201
    OP3 = 202

    # Sugar backbone
    O5p = 203
    C5p = 204
    C4p = 205
    O4p = 206
    C3p = 207
    O3p = 208
    C2p = 209
    O2p = 210
    C1p = 211

    # Nucleobase (cytosine)
    N1 = 212
    C2 = 213
    O2 = 214
    N3 = 215
    C4 = 216
    N4 = 217
    C5 = 218
    C6 = 219

    # Hydrogens
    HOC2 = 220
    HOP2 = 221
    HOP3 = 222
    H5p = 223
    H5pp = 224
    H4p = 225
    H3p = 226
    H2p = 227
    H1p = 228
    H41 = 229
    H42 = 230
    H5 = 231
    H6 = 232


class Deoxyguanosine(IndexEnum):
    """
    2'-Deoxyguanosine (GNG) atom indices.

    GNG is deoxyguanosine (missing 2'-OH).
    Indices start at 233.
    """

    # Phosphate group
    P = 233
    OP1 = 234
    OP2 = 235
    OP3 = 236

    # Sugar backbone (no O2')
    O5p = 237
    C5p = 238
    C4p = 239
    O4p = 240
    C3p = 241
    O3p = 242
    C2p = 243  # No O2' (deoxy)
    C1p = 244

    # Nucleobase (guanine)
    N9 = 245
    C8 = 246
    N7 = 247
    C5 = 248
    C6 = 249
    O6 = 250
    N1 = 251
    C2 = 252
    N2 = 253
    N3 = 254
    C4 = 255

    # Hydrogens
    HOP2 = 256
    HOP3 = 257
    H5p = 258
    H5pp = 259
    H4p = 260
    H3p = 261
    HO3p = 262
    H2p = 263
    H2pp = 264  # Second H on C2' (deoxy)
    H1p = 265
    H8 = 266
    H1 = 267
    H21 = 268
    H22 = 269


# Combined RNA nucleotide enum with prefixed names
RibonucleicAcid = IndexEnum(
    "RibonucleicAcid",
    Adenosine.dict("A_") | Cytosine.dict("C_") |
    Guanosine.dict("G_") | Uridine.dict("U_")
)

# Combined RNA nucleotide enum without prefixes
RibonucleicAcidNoPrefix = IndexEnum(
    "RibonucleicAcid",
    Adenosine.dict() | Cytosine.dict() |
    Guanosine.dict() | Uridine.dict()
)

# Modified nucleotides with prefixed names
ModifiedNucleotides = IndexEnum(
    "ModifiedNucleotides",
    GuanosineTriphosphate.dict("GTP_") |
    CytidineTriphosphate.dict("CCC_") |
    Deoxyguanosine.dict("GNG_")
)
