/* ANSI-C code produced by gperf version 3.3 */
/* Command-line: gperf atom.gperf  */
/* Computed positions: -k'1,3-6' */

#if !((' ' == 32) && ('!' == 33) && ('"' == 34) && ('#' == 35) \
      && ('%' == 37) && ('&' == 38) && ('\'' == 39) && ('(' == 40) \
      && (')' == 41) && ('*' == 42) && ('+' == 43) && (',' == 44) \
      && ('-' == 45) && ('.' == 46) && ('/' == 47) && ('0' == 48) \
      && ('1' == 49) && ('2' == 50) && ('3' == 51) && ('4' == 52) \
      && ('5' == 53) && ('6' == 54) && ('7' == 55) && ('8' == 56) \
      && ('9' == 57) && (':' == 58) && (';' == 59) && ('<' == 60) \
      && ('=' == 61) && ('>' == 62) && ('?' == 63) && ('A' == 65) \
      && ('B' == 66) && ('C' == 67) && ('D' == 68) && ('E' == 69) \
      && ('F' == 70) && ('G' == 71) && ('H' == 72) && ('I' == 73) \
      && ('J' == 74) && ('K' == 75) && ('L' == 76) && ('M' == 77) \
      && ('N' == 78) && ('O' == 79) && ('P' == 80) && ('Q' == 81) \
      && ('R' == 82) && ('S' == 83) && ('T' == 84) && ('U' == 85) \
      && ('V' == 86) && ('W' == 87) && ('X' == 88) && ('Y' == 89) \
      && ('Z' == 90) && ('[' == 91) && ('\\' == 92) && (']' == 93) \
      && ('^' == 94) && ('_' == 95) && ('a' == 97) && ('b' == 98) \
      && ('c' == 99) && ('d' == 100) && ('e' == 101) && ('f' == 102) \
      && ('g' == 103) && ('h' == 104) && ('i' == 105) && ('j' == 106) \
      && ('k' == 107) && ('l' == 108) && ('m' == 109) && ('n' == 110) \
      && ('o' == 111) && ('p' == 112) && ('q' == 113) && ('r' == 114) \
      && ('s' == 115) && ('t' == 116) && ('u' == 117) && ('v' == 118) \
      && ('w' == 119) && ('x' == 120) && ('y' == 121) && ('z' == 122) \
      && ('{' == 123) && ('|' == 124) && ('}' == 125) && ('~' == 126))
/* The character set is not based on ISO-646.  */
#error "gperf generated tables don't work with this execution character set. Please report a bug to <bug-gperf@gnu.org>."
#endif

#line 5 "atom.gperf"

#include "lookup.h"
#line 8 "atom.gperf"
struct _LOOKUP;

#define ATOMTOTAL_KEYWORDS 148
#define ATOMMIN_WORD_LENGTH 3
#define ATOMMAX_WORD_LENGTH 6
#define ATOMMIN_HASH_VALUE 13
#define ATOMMAX_HASH_VALUE 441
/* maximum key range = 429, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
static unsigned int
_hash_atom (register const char *str, register size_t len)
{
  static unsigned short asso_values[] =
    {
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442,   0,
      442, 442, 442, 442, 442, 442, 442, 442, 442,   3,
       45, 200, 214, 125, 104,  15,   0,   0, 442, 442,
      442, 442, 442, 442, 442,  25, 442,   5, 442, 442,
      442,  35,   0, 442, 442, 442, 442, 442,  70,  40,
      160, 442, 442, 442, 442,  15, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442, 442, 442, 442, 442,
      442, 442, 442, 442, 442, 442
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[5]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 5:
        hval += asso_values[(unsigned char)str[4]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

struct _LOOKUP *
_lookup_atom (register const char *str, register size_t len)
{
#if (defined __GNUC__ && __GNUC__ + (__GNUC_MINOR__ >= 6) > 4) || (defined __clang__ && __clang_major__ >= 3)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
  static struct _LOOKUP wordlist[] =
    {
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""},
#line 79 "atom.gperf"
      {"C_H1'", 68},
      {""}, {""}, {""}, {""},
#line 60 "atom.gperf"
      {"C_C1'", 50},
      {""}, {""}, {""}, {""},
#line 154 "atom.gperf"
      {"U_H1'", 141},
      {""}, {""}, {""}, {""},
#line 135 "atom.gperf"
      {"U_C1'", 123},
#line 44 "atom.gperf"
      {"A_H8", 34},
      {""}, {""}, {""},
#line 43 "atom.gperf"
      {"A_H1'", 33},
#line 24 "atom.gperf"
      {"A_C8", 15},
      {""}, {""}, {""},
#line 22 "atom.gperf"
      {"A_C1'", 13},
#line 119 "atom.gperf"
      {"G_H8", 107},
      {""}, {""},
#line 120 "atom.gperf"
      {"G_H1", 108},
#line 118 "atom.gperf"
      {"G_H1'", 106},
#line 98 "atom.gperf"
      {"G_C8", 87},
      {""}, {""}, {""},
#line 96 "atom.gperf"
      {"G_C1'", 85},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 76 "atom.gperf"
      {"C_H2'", 66},
      {""}, {""}, {""},
#line 62 "atom.gperf"
      {"C_C2", 52},
#line 58 "atom.gperf"
      {"C_C2'", 48},
      {""}, {""}, {""}, {""},
#line 151 "atom.gperf"
      {"U_H2'", 139},
      {""}, {""}, {""},
#line 137 "atom.gperf"
      {"U_C2", 125},
#line 133 "atom.gperf"
      {"U_C2'", 121},
      {""}, {""}, {""},
#line 47 "atom.gperf"
      {"A_H2", 37},
#line 40 "atom.gperf"
      {"A_H2'", 31},
      {""}, {""}, {""},
#line 30 "atom.gperf"
      {"A_C2", 21},
#line 20 "atom.gperf"
      {"A_C2'", 11},
      {""},
#line 61 "atom.gperf"
      {"C_N1", 51},
      {""}, {""},
#line 115 "atom.gperf"
      {"G_H2'", 104},
      {""}, {""},
#line 121 "atom.gperf"
      {"G_H21", 109},
#line 104 "atom.gperf"
      {"G_C2", 93},
#line 94 "atom.gperf"
      {"G_C2'", 83},
      {""},
#line 136 "atom.gperf"
      {"U_N1", 124},
      {""},
#line 63 "atom.gperf"
      {"C_O2", 53},
#line 59 "atom.gperf"
      {"C_O2'", 49},
#line 77 "atom.gperf"
      {"C_HO2'", 67},
      {""}, {""},
#line 23 "atom.gperf"
      {"A_N9", 14},
      {""}, {""},
#line 29 "atom.gperf"
      {"A_N1", 20},
      {""},
#line 138 "atom.gperf"
      {"U_O2", 126},
#line 134 "atom.gperf"
      {"U_O2'", 122},
#line 152 "atom.gperf"
      {"U_HO2'", 140},
      {""}, {""},
#line 97 "atom.gperf"
      {"G_N9", 86},
      {""}, {""},
#line 103 "atom.gperf"
      {"G_N1", 92},
#line 83 "atom.gperf"
      {"C_H6", 72},
#line 25 "atom.gperf"
      {"A_N7", 16},
#line 21 "atom.gperf"
      {"A_O2'", 12},
#line 41 "atom.gperf"
      {"A_HO2'", 32},
      {""},
#line 68 "atom.gperf"
      {"C_C6", 58},
      {""}, {""}, {""}, {""},
#line 157 "atom.gperf"
      {"U_H6", 144},
#line 99 "atom.gperf"
      {"G_N7", 88},
#line 95 "atom.gperf"
      {"G_O2'", 84},
#line 116 "atom.gperf"
      {"G_HO2'", 105},
      {""},
#line 143 "atom.gperf"
      {"U_C6", 131},
      {""},
#line 122 "atom.gperf"
      {"G_H22", 110},
      {""}, {""}, {""},
#line 82 "atom.gperf"
      {"C_H5", 71},
#line 71 "atom.gperf"
      {"C_H5'", 61},
#line 72 "atom.gperf"
      {"C_H5''", 62},
#line 45 "atom.gperf"
      {"A_H61", 35},
#line 27 "atom.gperf"
      {"A_C6", 18},
#line 67 "atom.gperf"
      {"C_C5", 57},
#line 53 "atom.gperf"
      {"C_C5'", 43},
      {""}, {""}, {""},
#line 156 "atom.gperf"
      {"U_H5", 143},
#line 146 "atom.gperf"
      {"U_H5'", 134},
#line 147 "atom.gperf"
      {"U_H5''", 135},
      {""},
#line 101 "atom.gperf"
      {"G_C6", 90},
#line 142 "atom.gperf"
      {"U_C5", 130},
#line 128 "atom.gperf"
      {"U_C5'", 116},
      {""}, {""}, {""},
#line 105 "atom.gperf"
      {"G_N2", 94},
#line 35 "atom.gperf"
      {"A_H5'", 26},
#line 36 "atom.gperf"
      {"A_H5''", 27},
      {""}, {""},
#line 26 "atom.gperf"
      {"A_C5", 17},
#line 15 "atom.gperf"
      {"A_C5'", 6},
      {""}, {""}, {""}, {""},
#line 111 "atom.gperf"
      {"G_H5'", 100},
#line 110 "atom.gperf"
      {"G_H5''", 99},
      {""},
#line 49 "atom.gperf"
      {"C_P", 39},
#line 100 "atom.gperf"
      {"G_C5", 89},
#line 89 "atom.gperf"
      {"G_C5'", 78},
      {""}, {""}, {""}, {""},
#line 52 "atom.gperf"
      {"C_O5'", 42},
#line 78 "atom.gperf"
      {"C_HO5'", 145},
      {""},
#line 124 "atom.gperf"
      {"U_P", 112},
#line 46 "atom.gperf"
      {"A_H62", 36},
      {""}, {""}, {""},
#line 102 "atom.gperf"
      {"G_O6", 91},
      {""},
#line 127 "atom.gperf"
      {"U_O5'", 115},
#line 153 "atom.gperf"
      {"U_HO5'", 147},
      {""},
#line 11 "atom.gperf"
      {"A_P", 2},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 14 "atom.gperf"
      {"A_O5'", 5},
#line 42 "atom.gperf"
      {"A_HO5'", 148},
      {""},
#line 85 "atom.gperf"
      {"G_P", 74},
      {""}, {""}, {""}, {""},
#line 28 "atom.gperf"
      {"A_N6", 19},
      {""},
#line 88 "atom.gperf"
      {"G_O5'", 77},
#line 117 "atom.gperf"
      {"G_HO5'", 146},
      {""}, {""}, {""},
#line 74 "atom.gperf"
      {"C_H3'", 64},
      {""}, {""},
#line 50 "atom.gperf"
      {"C_OP1", 40},
      {""},
#line 56 "atom.gperf"
      {"C_C3'", 46},
      {""}, {""}, {""},
#line 155 "atom.gperf"
      {"U_H3", 142},
#line 149 "atom.gperf"
      {"U_H3'", 137},
      {""}, {""},
#line 125 "atom.gperf"
      {"U_OP1", 113},
#line 73 "atom.gperf"
      {"C_H4'", 63},
#line 131 "atom.gperf"
      {"U_C3'", 119},
      {""},
#line 80 "atom.gperf"
      {"C_H41", 69},
#line 65 "atom.gperf"
      {"C_C4", 55},
#line 54 "atom.gperf"
      {"C_C4'", 44},
#line 38 "atom.gperf"
      {"A_H3'", 29},
      {""}, {""},
#line 12 "atom.gperf"
      {"A_OP1", 3},
#line 148 "atom.gperf"
      {"U_H4'", 136},
#line 18 "atom.gperf"
      {"A_C3'", 9},
      {""}, {""},
#line 140 "atom.gperf"
      {"U_C4", 128},
#line 129 "atom.gperf"
      {"U_C4'", 117},
#line 113 "atom.gperf"
      {"G_H3'", 102},
      {""}, {""},
#line 86 "atom.gperf"
      {"G_OP1", 75},
#line 37 "atom.gperf"
      {"A_H4'", 28},
#line 92 "atom.gperf"
      {"G_C3'", 81},
      {""}, {""},
#line 32 "atom.gperf"
      {"A_C4", 23},
#line 16 "atom.gperf"
      {"A_C4'", 7},
#line 57 "atom.gperf"
      {"C_O3'", 47},
#line 75 "atom.gperf"
      {"C_HO3'", 65},
      {""}, {""},
#line 112 "atom.gperf"
      {"G_H4'", 101},
#line 51 "atom.gperf"
      {"C_OP2", 41},
#line 70 "atom.gperf"
      {"C_HOP2", 60},
      {""},
#line 107 "atom.gperf"
      {"G_C4", 96},
#line 90 "atom.gperf"
      {"G_C4'", 79},
#line 132 "atom.gperf"
      {"U_O3'", 120},
#line 150 "atom.gperf"
      {"U_HO3'", 138},
      {""}, {""},
#line 55 "atom.gperf"
      {"C_O4'", 45},
#line 126 "atom.gperf"
      {"U_OP2", 114},
#line 145 "atom.gperf"
      {"U_HOP2", 133},
      {""}, {""},
#line 81 "atom.gperf"
      {"C_H42", 70},
#line 19 "atom.gperf"
      {"A_O3'", 10},
#line 39 "atom.gperf"
      {"A_HO3'", 30},
      {""},
#line 141 "atom.gperf"
      {"U_O4", 129},
#line 130 "atom.gperf"
      {"U_O4'", 118},
#line 13 "atom.gperf"
      {"A_OP2", 4},
#line 34 "atom.gperf"
      {"A_HOP2", 25},
      {""}, {""},
#line 64 "atom.gperf"
      {"C_N3", 54},
#line 93 "atom.gperf"
      {"G_O3'", 82},
#line 114 "atom.gperf"
      {"G_HO3'", 103},
      {""}, {""},
#line 17 "atom.gperf"
      {"A_O4'", 8},
#line 87 "atom.gperf"
      {"G_OP2", 76},
#line 109 "atom.gperf"
      {"G_HOP2", 98},
      {""}, {""},
#line 139 "atom.gperf"
      {"U_N3", 127},
      {""}, {""}, {""},
#line 66 "atom.gperf"
      {"C_N4", 56},
#line 91 "atom.gperf"
      {"G_O4'", 80},
      {""}, {""}, {""}, {""},
#line 31 "atom.gperf"
      {"A_N3", 22},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 106 "atom.gperf"
      {"G_N3", 95},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""},
#line 48 "atom.gperf"
      {"C_OP3", 38},
#line 69 "atom.gperf"
      {"C_HOP3", 59},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 123 "atom.gperf"
      {"U_OP3", 111},
#line 144 "atom.gperf"
      {"U_HOP3", 132},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 10 "atom.gperf"
      {"A_OP3", 1},
#line 33 "atom.gperf"
      {"A_HOP3", 24},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 84 "atom.gperf"
      {"G_OP3", 73},
#line 108 "atom.gperf"
      {"G_HOP3", 97}
    };
#if (defined __GNUC__ && __GNUC__ + (__GNUC_MINOR__ >= 6) > 4) || (defined __clang__ && __clang_major__ >= 3)
#pragma GCC diagnostic pop
#endif

  if (len <= ATOMMAX_WORD_LENGTH && len >= ATOMMIN_WORD_LENGTH)
    {
      register unsigned int key = _hash_atom (str, len);

      if (key <= ATOMMAX_HASH_VALUE)
        {
          register const char *s = wordlist[key].name;

          if (*str == *s && !strcmp (str + 1, s + 1))
            return &wordlist[key];
        }
    }
  return (struct _LOOKUP *) 0;
}
