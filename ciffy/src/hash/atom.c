/* ANSI-C code produced by gperf version 3.3 */
/* Command-line: gperf atom.gperf  */
/* Computed positions: -k'1,3-8' */

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

#define ATOMTOTAL_KEYWORDS 269
#define ATOMMIN_WORD_LENGTH 3
#define ATOMMAX_WORD_LENGTH 8
#define ATOMMIN_HASH_VALUE 9
#define ATOMMAX_HASH_VALUE 1199
/* maximum key range = 1191, duplicates = 0 */

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
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,    5,
       180, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,  505,
        85,  155,  335,  225,    0,   15,  115,   10, 1200, 1200,
      1200, 1200, 1200, 1200, 1200,  465,   20,    5,   25, 1200,
      1200,   15,    0, 1200, 1200, 1200, 1200, 1200,  225,   30,
        30,    0, 1200, 1200, 1200,  275, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200,   20, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
      1200, 1200, 1200, 1200, 1200, 1200, 1200
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[7]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 7:
        hval += asso_values[(unsigned char)str[6]+1];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 6:
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
#line 83 "atom.gperf"
      {"C_H6", 72},
      {""}, {""}, {""}, {""},
#line 68 "atom.gperf"
      {"C_C6", 58},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 101 "atom.gperf"
      {"G_C6", 90},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""},
#line 241 "atom.gperf"
      {"CCC_H6", 232},
      {""},
#line 49 "atom.gperf"
      {"C_P", 39},
      {""}, {""},
#line 228 "atom.gperf"
      {"CCC_C6", 219},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 85 "atom.gperf"
      {"G_P", 74},
#line 102 "atom.gperf"
      {"G_O6", 91},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""},
#line 258 "atom.gperf"
      {"GNG_C6", 249},
      {""}, {""}, {""},
#line 208 "atom.gperf"
      {"CCC_P", 199},
      {""}, {""}, {""}, {""}, {""},
#line 205 "atom.gperf"
      {"CCC_PC", 196},
      {""}, {""}, {""}, {""},
#line 183 "atom.gperf"
      {"GTP_C6", 174},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 242 "atom.gperf"
      {"GNG_P", 233},
#line 259 "atom.gperf"
      {"GNG_O6", 250},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""},
#line 62 "atom.gperf"
      {"C_C2", 52},
#line 76 "atom.gperf"
      {"C_H2'", 66},
#line 184 "atom.gperf"
      {"GTP_O6", 175},
      {""}, {""}, {""},
#line 58 "atom.gperf"
      {"C_C2'", 48},
      {""}, {""}, {""},
#line 104 "atom.gperf"
      {"G_C2", 93},
#line 115 "atom.gperf"
      {"G_H2'", 104},
      {""}, {""}, {""}, {""},
#line 94 "atom.gperf"
      {"G_C2'", 83},
#line 158 "atom.gperf"
      {"GTP_PG", 149},
      {""}, {""}, {""}, {""},
#line 163 "atom.gperf"
      {"GTP_PB", 154},
      {""}, {""},
#line 63 "atom.gperf"
      {"C_O2", 53},
      {""},
#line 222 "atom.gperf"
      {"CCC_C2", 213},
      {""}, {""}, {""},
#line 59 "atom.gperf"
      {"C_O2'", 49},
#line 77 "atom.gperf"
      {"C_HO2'", 67},
      {""}, {""},
#line 119 "atom.gperf"
      {"G_H8", 107},
      {""}, {""}, {""}, {""},
#line 98 "atom.gperf"
      {"G_C8", 87},
#line 95 "atom.gperf"
      {"G_O2'", 84},
#line 116 "atom.gperf"
      {"G_HO2'", 105},
      {""}, {""}, {""}, {""},
#line 261 "atom.gperf"
      {"GNG_C2", 252},
      {""}, {""}, {""}, {""},
#line 223 "atom.gperf"
      {"CCC_O2", 214},
      {""},
#line 230 "atom.gperf"
      {"CCC_HOP2", 221},
      {""},
#line 51 "atom.gperf"
      {"C_OP2", 41},
#line 70 "atom.gperf"
      {"C_HOP2", 60},
      {""}, {""}, {""}, {""},
#line 186 "atom.gperf"
      {"GTP_C2", 177},
      {""}, {""}, {""},
#line 87 "atom.gperf"
      {"G_OP2", 76},
#line 109 "atom.gperf"
      {"G_HOP2", 98},
      {""}, {""}, {""},
#line 74 "atom.gperf"
      {"C_H3'", 64},
#line 275 "atom.gperf"
      {"GNG_H8", 266},
      {""},
#line 265 "atom.gperf"
      {"GNG_HOP2", 256},
      {""},
#line 56 "atom.gperf"
      {"C_C3'", 46},
#line 255 "atom.gperf"
      {"GNG_C8", 246},
#line 207 "atom.gperf"
      {"CCC_O2C", 198},
#line 229 "atom.gperf"
      {"CCC_HOC2", 220},
      {""},
#line 113 "atom.gperf"
      {"G_H3'", 102},
      {""},
#line 209 "atom.gperf"
      {"CCC_OP1", 200},
      {""}, {""},
#line 92 "atom.gperf"
      {"G_C3'", 81},
#line 201 "atom.gperf"
      {"GTP_H8", 192},
#line 160 "atom.gperf"
      {"GTP_O2G", 151},
#line 190 "atom.gperf"
      {"GTP_HOG2", 181},
      {""},
#line 122 "atom.gperf"
      {"G_H22", 110},
#line 180 "atom.gperf"
      {"GTP_C8", 171},
#line 165 "atom.gperf"
      {"GTP_O2B", 156},
#line 192 "atom.gperf"
      {"GTP_HOB2", 183},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 57 "atom.gperf"
      {"C_O3'", 47},
#line 75 "atom.gperf"
      {"C_HO3'", 65},
#line 243 "atom.gperf"
      {"GNG_OP1", 234},
      {""}, {""}, {""}, {""},
#line 169 "atom.gperf"
      {"GTP_O2A", 160},
      {""}, {""},
#line 93 "atom.gperf"
      {"G_O3'", 82},
#line 114 "atom.gperf"
      {"G_HO3'", 103},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""},
#line 231 "atom.gperf"
      {"CCC_HOP3", 222},
      {""},
#line 48 "atom.gperf"
      {"C_OP3", 38},
#line 69 "atom.gperf"
      {"C_HOP3", 59},
#line 277 "atom.gperf"
      {"GNG_H21", 268},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 82 "atom.gperf"
      {"C_H5", 71},
#line 84 "atom.gperf"
      {"G_OP3", 73},
#line 108 "atom.gperf"
      {"G_HOP3", 97},
      {""}, {""},
#line 67 "atom.gperf"
      {"C_C5", 57},
#line 71 "atom.gperf"
      {"C_H5'", 61},
      {""},
#line 203 "atom.gperf"
      {"GTP_H21", 194},
#line 266 "atom.gperf"
      {"GNG_HOP3", 257},
      {""},
#line 53 "atom.gperf"
      {"C_C5'", 43},
#line 72 "atom.gperf"
      {"C_H5''", 62},
      {""}, {""},
#line 100 "atom.gperf"
      {"G_C5", 89},
#line 111 "atom.gperf"
      {"G_H5'", 100},
      {""},
#line 210 "atom.gperf"
      {"CCC_OP2", 201},
      {""},
#line 97 "atom.gperf"
      {"G_N9", 86},
#line 89 "atom.gperf"
      {"G_C5'", 78},
#line 110 "atom.gperf"
      {"G_H5''", 99},
#line 161 "atom.gperf"
      {"GTP_O3G", 152},
#line 191 "atom.gperf"
      {"GTP_HOG3", 182},
#line 99 "atom.gperf"
      {"G_N7", 88},
      {""},
#line 240 "atom.gperf"
      {"CCC_H5", 231},
#line 162 "atom.gperf"
      {"GTP_O3B", 153},
#line 199 "atom.gperf"
      {"GTP_HO2'", 190},
      {""}, {""},
#line 227 "atom.gperf"
      {"CCC_C5", 218},
      {""}, {""}, {""},
#line 52 "atom.gperf"
      {"C_O5'", 42},
#line 78 "atom.gperf"
      {"C_HO5'", 145},
#line 244 "atom.gperf"
      {"GNG_OP2", 235},
      {""}, {""}, {""}, {""},
#line 166 "atom.gperf"
      {"GTP_O3A", 157},
      {""},
#line 157 "atom.gperf"
      {"U_H6", 144},
#line 88 "atom.gperf"
      {"G_O5'", 77},
#line 117 "atom.gperf"
      {"G_HO5'", 146},
      {""}, {""},
#line 143 "atom.gperf"
      {"U_C6", 131},
      {""},
#line 257 "atom.gperf"
      {"GNG_C5", 248},
      {""}, {""}, {""}, {""},
#line 254 "atom.gperf"
      {"GNG_N9", 245},
      {""}, {""}, {""}, {""},
#line 256 "atom.gperf"
      {"GNG_N7", 247},
#line 278 "atom.gperf"
      {"GNG_H22", 269},
      {""}, {""}, {""},
#line 182 "atom.gperf"
      {"GTP_C5", 173},
#line 236 "atom.gperf"
      {"CCC_H2'", 227},
      {""}, {""}, {""},
#line 179 "atom.gperf"
      {"GTP_N9", 170},
#line 218 "atom.gperf"
      {"CCC_C2'", 209},
#line 124 "atom.gperf"
      {"U_P", 112},
      {""}, {""},
#line 181 "atom.gperf"
      {"GTP_N7", 172},
#line 204 "atom.gperf"
      {"GTP_H22", 195},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 272 "atom.gperf"
      {"GNG_H2'", 263},
      {""}, {""}, {""}, {""},
#line 252 "atom.gperf"
      {"GNG_C2'", 243},
#line 273 "atom.gperf"
      {"GNG_H2''", 264},
#line 105 "atom.gperf"
      {"G_N2", 94},
      {""}, {""},
#line 219 "atom.gperf"
      {"CCC_O2'", 210},
      {""}, {""}, {""}, {""},
#line 198 "atom.gperf"
      {"GTP_H2'", 189},
      {""}, {""}, {""}, {""},
#line 176 "atom.gperf"
      {"GTP_C2'", 167},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 65 "atom.gperf"
      {"C_C4", 55},
#line 73 "atom.gperf"
      {"C_H4'", 63},
      {""}, {""}, {""}, {""},
#line 54 "atom.gperf"
      {"C_C4'", 44},
      {""}, {""}, {""},
#line 107 "atom.gperf"
      {"G_C4", 96},
#line 112 "atom.gperf"
      {"G_H4'", 101},
      {""}, {""}, {""}, {""},
#line 90 "atom.gperf"
      {"G_C4'", 79},
#line 262 "atom.gperf"
      {"GNG_N2", 253},
#line 177 "atom.gperf"
      {"GTP_O2'", 168},
      {""},
#line 137 "atom.gperf"
      {"U_C2", 125},
#line 151 "atom.gperf"
      {"U_H2'", 139},
      {""},
#line 235 "atom.gperf"
      {"CCC_H3'", 226},
      {""}, {""},
#line 133 "atom.gperf"
      {"U_C2'", 121},
#line 225 "atom.gperf"
      {"CCC_C4", 216},
#line 216 "atom.gperf"
      {"CCC_C3'", 207},
      {""}, {""},
#line 55 "atom.gperf"
      {"C_O4'", 45},
#line 187 "atom.gperf"
      {"GTP_N2", 178},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 64 "atom.gperf"
      {"C_N3", 54},
#line 91 "atom.gperf"
      {"G_O4'", 80},
      {""},
#line 270 "atom.gperf"
      {"GNG_H3'", 261},
      {""},
#line 138 "atom.gperf"
      {"U_O2", 126},
      {""},
#line 264 "atom.gperf"
      {"GNG_C4", 255},
#line 250 "atom.gperf"
      {"GNG_C3'", 241},
      {""},
#line 106 "atom.gperf"
      {"G_N3", 95},
#line 134 "atom.gperf"
      {"U_O2'", 122},
#line 152 "atom.gperf"
      {"U_HO2'", 140},
#line 217 "atom.gperf"
      {"CCC_O3'", 208},
      {""}, {""}, {""}, {""},
#line 196 "atom.gperf"
      {"GTP_H3'", 187},
      {""}, {""}, {""},
#line 189 "atom.gperf"
      {"GTP_C4", 180},
#line 174 "atom.gperf"
      {"GTP_C3'", 165},
      {""}, {""}, {""},
#line 224 "atom.gperf"
      {"CCC_N3", 215},
      {""}, {""}, {""}, {""}, {""},
#line 251 "atom.gperf"
      {"GNG_O3'", 242},
      {""}, {""},
#line 126 "atom.gperf"
      {"U_OP2", 114},
#line 145 "atom.gperf"
      {"U_HOP2", 133},
      {""},
#line 271 "atom.gperf"
      {"GNG_HO3'", 262},
      {""},
#line 81 "atom.gperf"
      {"C_H42", 70},
      {""},
#line 211 "atom.gperf"
      {"CCC_OP3", 202},
      {""},
#line 155 "atom.gperf"
      {"U_H3", 142},
      {""},
#line 263 "atom.gperf"
      {"GNG_N3", 254},
#line 175 "atom.gperf"
      {"GTP_O3'", 166},
      {""}, {""},
#line 149 "atom.gperf"
      {"U_H3'", 137},
      {""},
#line 232 "atom.gperf"
      {"CCC_H5'", 223},
#line 197 "atom.gperf"
      {"GTP_HO3'", 188},
      {""},
#line 131 "atom.gperf"
      {"U_C3'", 119},
      {""},
#line 213 "atom.gperf"
      {"CCC_C5'", 204},
#line 233 "atom.gperf"
      {"CCC_H5''", 224},
      {""}, {""},
#line 188 "atom.gperf"
      {"GTP_N3", 179},
#line 245 "atom.gperf"
      {"GNG_OP3", 236},
      {""}, {""}, {""}, {""},
#line 238 "atom.gperf"
      {"CCC_H41", 229},
      {""}, {""}, {""}, {""},
#line 267 "atom.gperf"
      {"GNG_H5'", 258},
      {""}, {""}, {""}, {""},
#line 247 "atom.gperf"
      {"GNG_C5'", 238},
#line 268 "atom.gperf"
      {"GNG_H5''", 259},
      {""},
#line 132 "atom.gperf"
      {"U_O3'", 120},
#line 150 "atom.gperf"
      {"U_HO3'", 138},
#line 212 "atom.gperf"
      {"CCC_O5'", 203},
      {""},
#line 27 "atom.gperf"
      {"A_C6", 18},
      {""}, {""},
#line 193 "atom.gperf"
      {"GTP_H5'", 184},
      {""}, {""}, {""}, {""},
#line 171 "atom.gperf"
      {"GTP_C5'", 162},
#line 194 "atom.gperf"
      {"GTP_H5''", 185},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 246 "atom.gperf"
      {"GNG_O5'", 237},
      {""}, {""},
#line 123 "atom.gperf"
      {"U_OP3", 111},
#line 144 "atom.gperf"
      {"U_HOP3", 132},
      {""},
#line 11 "atom.gperf"
      {"A_P", 2},
      {""}, {""}, {""}, {""}, {""},
#line 156 "atom.gperf"
      {"U_H5", 143},
      {""}, {""},
#line 170 "atom.gperf"
      {"GTP_O5'", 161},
      {""},
#line 142 "atom.gperf"
      {"U_C5", 130},
#line 146 "atom.gperf"
      {"U_H5'", 134},
      {""}, {""}, {""}, {""},
#line 128 "atom.gperf"
      {"U_C5'", 116},
#line 147 "atom.gperf"
      {"U_H5''", 135},
      {""}, {""}, {""},
#line 79 "atom.gperf"
      {"C_H1'", 68},
      {""}, {""}, {""},
#line 120 "atom.gperf"
      {"G_H1", 108},
#line 60 "atom.gperf"
      {"C_C1'", 50},
      {""},
#line 239 "atom.gperf"
      {"CCC_H42", 230},
      {""}, {""},
#line 118 "atom.gperf"
      {"G_H1'", 106},
      {""}, {""}, {""}, {""},
#line 96 "atom.gperf"
      {"G_C1'", 85},
      {""}, {""}, {""}, {""},
#line 127 "atom.gperf"
      {"U_O5'", 115},
#line 153 "atom.gperf"
      {"U_HO5'", 147},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""},
#line 234 "atom.gperf"
      {"CCC_H4'", 225},
      {""},
#line 47 "atom.gperf"
      {"A_H2", 37},
#line 46 "atom.gperf"
      {"A_H62", 36},
      {""},
#line 214 "atom.gperf"
      {"CCC_C4'", 205},
      {""},
#line 30 "atom.gperf"
      {"A_C2", 21},
#line 40 "atom.gperf"
      {"A_H2'", 31},
#line 276 "atom.gperf"
      {"GNG_H1", 267},
      {""}, {""}, {""},
#line 20 "atom.gperf"
      {"A_C2'", 11},
#line 167 "atom.gperf"
      {"GTP_PA", 158},
      {""}, {""},
#line 66 "atom.gperf"
      {"C_N4", 56},
      {""}, {""},
#line 269 "atom.gperf"
      {"GNG_H4'", 260},
      {""}, {""},
#line 50 "atom.gperf"
      {"C_OP1", 40},
#line 202 "atom.gperf"
      {"GTP_H1", 193},
#line 248 "atom.gperf"
      {"GNG_C4'", 239},
      {""}, {""}, {""}, {""},
#line 215 "atom.gperf"
      {"CCC_O4'", 206},
      {""},
#line 44 "atom.gperf"
      {"A_H8", 34},
#line 86 "atom.gperf"
      {"G_OP1", 75},
      {""},
#line 195 "atom.gperf"
      {"GTP_H4'", 186},
      {""},
#line 24 "atom.gperf"
      {"A_C8", 15},
#line 21 "atom.gperf"
      {"A_O2'", 12},
#line 41 "atom.gperf"
      {"A_HO2'", 32},
#line 172 "atom.gperf"
      {"GTP_C4'", 163},
      {""}, {""}, {""},
#line 226 "atom.gperf"
      {"CCC_N4", 217},
#line 206 "atom.gperf"
      {"CCC_O1C", 197},
      {""}, {""}, {""}, {""},
#line 249 "atom.gperf"
      {"GNG_O4'", 240},
      {""}, {""}, {""}, {""},
#line 159 "atom.gperf"
      {"GTP_O1G", 150},
      {""}, {""},
#line 121 "atom.gperf"
      {"G_H21", 109},
      {""},
#line 164 "atom.gperf"
      {"GTP_O1B", 155},
      {""}, {""},
#line 13 "atom.gperf"
      {"A_OP2", 4},
#line 34 "atom.gperf"
      {"A_HOP2", 25},
#line 173 "atom.gperf"
      {"GTP_O4'", 164},
      {""},
#line 140 "atom.gperf"
      {"U_C4", 128},
#line 148 "atom.gperf"
      {"U_H4'", 136},
      {""}, {""}, {""}, {""},
#line 129 "atom.gperf"
      {"U_C4'", 117},
      {""},
#line 168 "atom.gperf"
      {"GTP_O1A", 159},
      {""}, {""},
#line 38 "atom.gperf"
      {"A_H3'", 29},
      {""}, {""}, {""}, {""},
#line 18 "atom.gperf"
      {"A_C3'", 9},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 141 "atom.gperf"
      {"U_O4", 129},
      {""}, {""}, {""}, {""}, {""},
#line 130 "atom.gperf"
      {"U_O4'", 118},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 139 "atom.gperf"
      {"U_N3", 127},
#line 19 "atom.gperf"
      {"A_O3'", 10},
#line 39 "atom.gperf"
      {"A_HO3'", 30},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""},
#line 10 "atom.gperf"
      {"A_OP3", 1},
#line 33 "atom.gperf"
      {"A_HOP3", 24},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 28 "atom.gperf"
      {"A_N6", 19},
      {""}, {""}, {""}, {""},
#line 26 "atom.gperf"
      {"A_C5", 17},
#line 35 "atom.gperf"
      {"A_H5'", 26},
      {""}, {""}, {""},
#line 23 "atom.gperf"
      {"A_N9", 14},
#line 15 "atom.gperf"
      {"A_C5'", 6},
#line 36 "atom.gperf"
      {"A_H5''", 27},
      {""}, {""},
#line 25 "atom.gperf"
      {"A_N7", 16},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""},
#line 237 "atom.gperf"
      {"CCC_H1'", 228},
      {""}, {""}, {""}, {""},
#line 220 "atom.gperf"
      {"CCC_C1'", 211},
      {""}, {""},
#line 14 "atom.gperf"
      {"A_O5'", 5},
#line 42 "atom.gperf"
      {"A_HO5'", 148},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 61 "atom.gperf"
      {"C_N1", 51},
      {""}, {""},
#line 274 "atom.gperf"
      {"GNG_H1'", 265},
      {""}, {""}, {""}, {""},
#line 253 "atom.gperf"
      {"GNG_C1'", 244},
      {""},
#line 103 "atom.gperf"
      {"G_N1", 92},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 200 "atom.gperf"
      {"GTP_H1'", 191},
      {""}, {""}, {""}, {""},
#line 178 "atom.gperf"
      {"GTP_C1'", 169},
      {""}, {""}, {""},
#line 221 "atom.gperf"
      {"CCC_N1", 212},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""},
#line 260 "atom.gperf"
      {"GNG_N1", 251},
      {""}, {""}, {""},
#line 154 "atom.gperf"
      {"U_H1'", 141},
      {""}, {""}, {""}, {""},
#line 135 "atom.gperf"
      {"U_C1'", 123},
      {""}, {""}, {""}, {""}, {""},
#line 185 "atom.gperf"
      {"GTP_N1", 176},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 32 "atom.gperf"
      {"A_C4", 23},
#line 37 "atom.gperf"
      {"A_H4'", 28},
      {""}, {""}, {""}, {""},
#line 16 "atom.gperf"
      {"A_C4'", 7},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 17 "atom.gperf"
      {"A_O4'", 8},
      {""}, {""}, {""}, {""},
#line 125 "atom.gperf"
      {"U_OP1", 113},
      {""}, {""}, {""},
#line 31 "atom.gperf"
      {"A_N3", 22},
#line 80 "atom.gperf"
      {"C_H41", 69},
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
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 45 "atom.gperf"
      {"A_H61", 35},
      {""}, {""}, {""}, {""},
#line 43 "atom.gperf"
      {"A_H1'", 33},
      {""}, {""}, {""}, {""},
#line 22 "atom.gperf"
      {"A_C1'", 13},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""},
#line 136 "atom.gperf"
      {"U_N1", 124},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 12 "atom.gperf"
      {"A_OP1", 3},
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
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""},
#line 29 "atom.gperf"
      {"A_N1", 20}
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
