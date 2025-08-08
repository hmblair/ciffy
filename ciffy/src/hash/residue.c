/* ANSI-C code produced by gperf version 3.3 */
/* Command-line: gperf residue.gperf  */
/* Computed positions: -k'1-3' */

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

#line 5 "residue.gperf"

#include "lookup.h"
#line 8 "residue.gperf"
struct _LOOKUP;

#define RESIDUETOTAL_KEYWORDS 30
#define RESIDUEMIN_WORD_LENGTH 1
#define RESIDUEMAX_WORD_LENGTH 3
#define RESIDUEMIN_HASH_VALUE 1
#define RESIDUEMAX_HASH_VALUE 66
/* maximum key range = 66, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
static unsigned int
_hash_residue (register const char *str, register size_t len)
{
  static unsigned char asso_values[] =
    {
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 10, 67, 15,  0, 23,
      67,  5, 15, 30, 67, 67,  0, 20, 28, 30,
      20, 67, 10, 10, 20,  0, 18, 67, 67,  5,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
      67, 67, 67, 67, 67, 67
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
#if (defined __cplusplus && (__cplusplus >= 201703L || (__cplusplus >= 201103L && defined __clang__ && __clang_major__ + (__clang_minor__ >= 9) > 3))) || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 202000L && ((defined __GNUC__ && __GNUC__ >= 10) || (defined __clang__ && __clang_major__ >= 9)))
      [[fallthrough]];
#elif (defined __GNUC__ && __GNUC__ >= 7) || (defined __clang__ && __clang_major__ >= 10)
      __attribute__ ((__fallthrough__));
#endif
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

struct _LOOKUP *
_lookup_residue (register const char *str, register size_t len)
{
#if (defined __GNUC__ && __GNUC__ + (__GNUC_MINOR__ >= 6) > 4) || (defined __clang__ && __clang_major__ >= 3)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
  static struct _LOOKUP wordlist[] =
    {
      {""},
#line 16 "residue.gperf"
      {"U", 3},
#line 17 "residue.gperf"
      {"DU", 3},
      {""}, {""}, {""},
#line 14 "residue.gperf"
      {"G", 2},
#line 15 "residue.gperf"
      {"DG", 2},
#line 23 "residue.gperf"
      {"GLU", 8},
      {""}, {""},
#line 10 "residue.gperf"
      {"A", 0},
#line 11 "residue.gperf"
      {"DA", 0},
#line 25 "residue.gperf"
      {"GLY", 10},
      {""}, {""},
#line 12 "residue.gperf"
      {"C", 1},
#line 13 "residue.gperf"
      {"DC", 1},
#line 28 "residue.gperf"
      {"LYS", 13},
      {""}, {""},
#line 18 "residue.gperf"
      {"T", 4},
#line 19 "residue.gperf"
      {"DT", 4},
#line 20 "residue.gperf"
      {"ALA", 5},
      {""}, {""},
#line 29 "residue.gperf"
      {"LEU", 14},
      {""},
#line 34 "residue.gperf"
      {"ARG", 19},
      {""}, {""},
#line 37 "residue.gperf"
      {"VAL", 22},
      {""},
#line 21 "residue.gperf"
      {"CYS", 6},
      {""}, {""},
#line 33 "residue.gperf"
      {"GLN", 18},
      {""},
#line 39 "residue.gperf"
      {"TYR", 24},
      {""}, {""}, {""}, {""},
#line 22 "residue.gperf"
      {"ASP", 7},
      {""}, {""},
#line 35 "residue.gperf"
      {"SER", 20},
      {""},
#line 36 "residue.gperf"
      {"THR", 21},
      {""}, {""},
#line 31 "residue.gperf"
      {"ASN", 16},
      {""},
#line 38 "residue.gperf"
      {"TRP", 23},
      {""}, {""},
#line 27 "residue.gperf"
      {"ILE", 12},
      {""},
#line 26 "residue.gperf"
      {"HIS", 11},
      {""}, {""},
#line 24 "residue.gperf"
      {"PHE", 9},
      {""},
#line 32 "residue.gperf"
      {"PRO", 17},
      {""}, {""},
#line 30 "residue.gperf"
      {"MET", 15}
    };
#if (defined __GNUC__ && __GNUC__ + (__GNUC_MINOR__ >= 6) > 4) || (defined __clang__ && __clang_major__ >= 3)
#pragma GCC diagnostic pop
#endif

  if (len <= RESIDUEMAX_WORD_LENGTH && len >= RESIDUEMIN_WORD_LENGTH)
    {
      register unsigned int key = _hash_residue (str, len);

      if (key <= RESIDUEMAX_HASH_VALUE)
        {
          register const char *s = wordlist[key].name;

          if (*str == *s && !strcmp (str + 1, s + 1))
            return &wordlist[key];
        }
    }
  return (struct _LOOKUP *) 0;
}
