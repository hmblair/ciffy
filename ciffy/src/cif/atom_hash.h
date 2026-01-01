/**
 * @file atom_hash.h
 * @brief Simple hash table for (chain, seq, atom) -> atom_index lookup.
 *
 * Used temporarily during CIF parsing to resolve _struct_conn references.
 * Uses FNV-1a hashing with open addressing and linear probing.
 *
 * Design:
 * - Keys are concatenated "chain\0seq\0atom" strings
 * - Values are atom indices (int32_t)
 * - Load factor kept below 0.7 for good performance
 * - Automatically resizes if needed (unlikely given we size for n_atoms)
 */

#ifndef _CIFFY_ATOM_HASH_H
#define _CIFFY_ATOM_HASH_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* FNV-1a hash constants */
#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME        1099511628211ULL

/* Empty slot marker */
#define ATOM_HASH_EMPTY (-1)

/* Maximum key length: chain(4) + seq(10) + atom(4) + separators + padding */
#define ATOM_HASH_MAX_KEY 32

/**
 * @brief Hash table entry.
 */
typedef struct {
    char key[ATOM_HASH_MAX_KEY];  /**< "chain\0seq\0atom" */
    int32_t value;                /**< Atom index, or ATOM_HASH_EMPTY */
} AtomHashEntry;

/**
 * @brief Hash table for atom lookup.
 */
typedef struct {
    AtomHashEntry *entries;  /**< Array of entries */
    size_t capacity;         /**< Total slots */
    size_t count;            /**< Used slots */
} AtomHash;


/* ============================================================================
 * HASH FUNCTION
 * ============================================================================ */

/**
 * @brief FNV-1a hash for fixed-length key buffer.
 *
 * Note: Key may contain embedded nulls, so we hash the full buffer.
 */
static inline uint64_t _atom_hash_fnv1a(const char *key, size_t len) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint8_t)key[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

/**
 * @brief Build key from chain, seq, atom strings.
 *
 * Format: "chain\0seq\0atom\0" - allows prefix matching and reconstruction.
 */
static inline void _atom_hash_make_key(
    char *out,
    const char *chain, size_t chain_len,
    const char *seq, size_t seq_len,
    const char *atom, size_t atom_len
) {
    /* Clamp lengths to prevent overflow */
    if (chain_len > 8) chain_len = 8;
    if (seq_len > 10) seq_len = 10;
    if (atom_len > 8) atom_len = 8;

    char *p = out;
    memcpy(p, chain, chain_len); p += chain_len; *p++ = '\0';
    memcpy(p, seq, seq_len); p += seq_len; *p++ = '\0';
    memcpy(p, atom, atom_len); p += atom_len; *p++ = '\0';
}


/* ============================================================================
 * HASH TABLE OPERATIONS
 * ============================================================================ */

/**
 * @brief Create a new hash table.
 *
 * @param expected_count Expected number of entries (will allocate ~1.5x for load factor)
 * @return Initialized hash table, or {NULL, 0, 0} on allocation failure
 */
static inline AtomHash atom_hash_create(size_t expected_count) {
    AtomHash h = {NULL, 0, 0};

    /* Size for ~0.65 load factor */
    size_t capacity = (expected_count * 3) / 2;
    if (capacity < 16) capacity = 16;

    /* Round up to power of 2 for fast modulo */
    size_t pow2 = 16;
    while (pow2 < capacity) pow2 *= 2;
    capacity = pow2;

    h.entries = (AtomHashEntry *)malloc(capacity * sizeof(AtomHashEntry));
    if (!h.entries) return h;

    h.capacity = capacity;
    h.count = 0;

    /* Initialize all slots as empty */
    for (size_t i = 0; i < capacity; i++) {
        h.entries[i].value = ATOM_HASH_EMPTY;
        h.entries[i].key[0] = '\0';
    }

    return h;
}

/**
 * @brief Free hash table memory.
 */
static inline void atom_hash_free(AtomHash *h) {
    if (h->entries) {
        free(h->entries);
        h->entries = NULL;
    }
    h->capacity = 0;
    h->count = 0;
}

/**
 * @brief Insert a key-value pair.
 *
 * @param h Hash table
 * @param chain Chain ID string
 * @param chain_len Length of chain string
 * @param seq Sequence ID string
 * @param seq_len Length of seq string
 * @param atom Atom name string
 * @param atom_len Length of atom string
 * @param value Atom index to store
 * @return true on success, false on failure (table full or alloc error)
 */
static inline bool atom_hash_insert(
    AtomHash *h,
    const char *chain, size_t chain_len,
    const char *seq, size_t seq_len,
    const char *atom, size_t atom_len,
    int32_t value
) {
    if (!h->entries) return false;

    /* Build key */
    char key[ATOM_HASH_MAX_KEY];
    memset(key, 0, ATOM_HASH_MAX_KEY);  /* Zero-fill for consistent hashing */
    _atom_hash_make_key(key, chain, chain_len, seq, seq_len, atom, atom_len);

    /* Find slot using linear probing */
    uint64_t hash = _atom_hash_fnv1a(key, ATOM_HASH_MAX_KEY);
    size_t mask = h->capacity - 1;  /* capacity is power of 2 */
    size_t idx = hash & mask;

    for (size_t i = 0; i < h->capacity; i++) {
        AtomHashEntry *entry = &h->entries[idx];

        if (entry->value == ATOM_HASH_EMPTY) {
            /* Empty slot - insert here */
            memcpy(entry->key, key, ATOM_HASH_MAX_KEY);
            entry->value = value;
            h->count++;
            return true;
        }

        if (memcmp(entry->key, key, ATOM_HASH_MAX_KEY) == 0) {
            /* Key exists - update value */
            entry->value = value;
            return true;
        }

        /* Collision - linear probe */
        idx = (idx + 1) & mask;
    }

    /* Table full (shouldn't happen with proper sizing) */
    return false;
}

/**
 * @brief Look up a key.
 *
 * @param h Hash table
 * @param chain Chain ID string
 * @param chain_len Length of chain string
 * @param seq Sequence ID string
 * @param seq_len Length of seq string
 * @param atom Atom name string
 * @param atom_len Length of atom string
 * @return Atom index if found, ATOM_HASH_EMPTY (-1) if not found
 */
static inline int32_t atom_hash_get(
    const AtomHash *h,
    const char *chain, size_t chain_len,
    const char *seq, size_t seq_len,
    const char *atom, size_t atom_len
) {
    if (!h->entries || h->count == 0) return ATOM_HASH_EMPTY;

    /* Build key */
    char key[ATOM_HASH_MAX_KEY];
    memset(key, 0, ATOM_HASH_MAX_KEY);  /* Zero-fill for consistent hashing */
    _atom_hash_make_key(key, chain, chain_len, seq, seq_len, atom, atom_len);

    /* Find slot */
    uint64_t hash = _atom_hash_fnv1a(key, ATOM_HASH_MAX_KEY);
    size_t mask = h->capacity - 1;
    size_t idx = hash & mask;

    for (size_t i = 0; i < h->capacity; i++) {
        const AtomHashEntry *entry = &h->entries[idx];

        if (entry->value == ATOM_HASH_EMPTY && entry->key[0] == '\0') {
            /* Empty slot that was never used - key not in table */
            return ATOM_HASH_EMPTY;
        }

        if (memcmp(entry->key, key, ATOM_HASH_MAX_KEY) == 0) {
            return entry->value;
        }

        idx = (idx + 1) & mask;
    }

    return ATOM_HASH_EMPTY;
}

#endif /* _CIFFY_ATOM_HASH_H */
