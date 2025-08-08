/* ANSI-C code produced by gperf version 3.3 */
/* Command-line: gperf atom.gperf  */
/* Computed positions: -k'1-4' */

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

#define ATOMTOTAL_KEYWORDS 51
#define ATOMMIN_WORD_LENGTH 1
#define ATOMMAX_WORD_LENGTH 4
#define ATOMMIN_HASH_VALUE 2
#define ATOMMAX_HASH_VALUE 125
/* maximum key range = 124, duplicates = 0 */

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
  static unsigned char asso_values[] =
    {
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126,   0,
      126, 126, 126, 126, 126, 126, 126, 126, 126,  55,
       15,  61,  35,   0,  27,  10,  50,   5, 126, 126,
      126, 126, 126, 126, 126, 126, 126,  10, 126, 126,
      126, 126,   0, 126, 126, 126, 126, 126,  60,   5,
       55, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
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
_lookup_atom (register const char *str, register size_t len)
{
#if (defined __GNUC__ && __GNUC__ + (__GNUC_MINOR__ >= 6) > 4) || (defined __clang__ && __clang_major__ >= 3)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
  static struct _LOOKUP wordlist[] =
    {
      {""}, {""},
#line 52 "atom.gperf"
      {"H5", 143},
#line 35 "atom.gperf"
      {"H5'", 134},
#line 36 "atom.gperf"
      {"H5''", 135},
      {""}, {""}, {""},
#line 14 "atom.gperf"
      {"O5'", 115},
#line 42 "atom.gperf"
      {"HO5'", 147},
      {""}, {""},
#line 26 "atom.gperf"
      {"C5", 130},
#line 15 "atom.gperf"
      {"C5'", 116},
      {""}, {""}, {""},
#line 47 "atom.gperf"
      {"H2", 37},
#line 40 "atom.gperf"
      {"H2'", 139},
      {""}, {""}, {""},
#line 48 "atom.gperf"
      {"O2", 126},
#line 21 "atom.gperf"
      {"O2'", 122},
#line 41 "atom.gperf"
      {"HO2'", 140},
      {""}, {""},
#line 30 "atom.gperf"
      {"C2", 125},
#line 20 "atom.gperf"
      {"C2'", 121},
#line 53 "atom.gperf"
      {"H6", 144},
      {""}, {""}, {""},
#line 58 "atom.gperf"
      {"H22", 110},
#line 54 "atom.gperf"
      {"O6", 91},
      {""}, {""}, {""},
#line 37 "atom.gperf"
      {"H4'", 136},
#line 27 "atom.gperf"
      {"C6", 131},
      {""}, {""},
#line 59 "atom.gperf"
      {"O4", 129},
#line 17 "atom.gperf"
      {"O4'", 118},
      {""},
#line 46 "atom.gperf"
      {"H62", 36},
      {""},
#line 32 "atom.gperf"
      {"C4", 128},
#line 16 "atom.gperf"
      {"C4'", 117},
      {""}, {""}, {""},
#line 44 "atom.gperf"
      {"H8", 107},
#line 51 "atom.gperf"
      {"H42", 70},
      {""}, {""},
#line 11 "atom.gperf"
      {"P", 112},
#line 56 "atom.gperf"
      {"H1", 108},
#line 43 "atom.gperf"
      {"H1'", 141},
      {""}, {""}, {""},
#line 24 "atom.gperf"
      {"C8", 87},
#line 60 "atom.gperf"
      {"H3", 142},
#line 38 "atom.gperf"
      {"H3'", 137},
      {""}, {""},
#line 23 "atom.gperf"
      {"N9", 86},
#line 22 "atom.gperf"
      {"C1'", 123},
#line 19 "atom.gperf"
      {"O3'", 120},
#line 39 "atom.gperf"
      {"HO3'", 138},
      {""},
#line 25 "atom.gperf"
      {"N7", 88},
#line 57 "atom.gperf"
      {"H21", 109},
#line 18 "atom.gperf"
      {"C3'", 119},
      {""}, {""},
#line 55 "atom.gperf"
      {"N2", 94},
#line 13 "atom.gperf"
      {"OP2", 114},
#line 34 "atom.gperf"
      {"HOP2", 133},
      {""}, {""}, {""}, {""}, {""},
#line 45 "atom.gperf"
      {"H61", 35},
      {""}, {""}, {""},
#line 28 "atom.gperf"
      {"N6", 19},
      {""}, {""}, {""},
#line 50 "atom.gperf"
      {"H41", 69},
      {""}, {""}, {""},
#line 49 "atom.gperf"
      {"N4", 56},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""},
#line 29 "atom.gperf"
      {"N1", 124},
#line 12 "atom.gperf"
      {"OP1", 113},
      {""}, {""}, {""}, {""},
#line 31 "atom.gperf"
      {"N3", 127},
#line 10 "atom.gperf"
      {"OP3", 111},
#line 33 "atom.gperf"
      {"HOP3", 132}
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
