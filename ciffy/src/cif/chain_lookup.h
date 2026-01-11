/**
 * @file chain_lookup.h
 * @brief Reusable chain name -> index hash table.
 *
 * Provides O(1) chain name lookups using FNV-1a hashing with linear probing.
 * Build once after cif->names is populated, use everywhere chain lookups
 * are needed.
 *
 * Usage:
 *   ChainLookup lookup;
 *   chain_lookup_build(&lookup, cif->names, cif->chains);
 *   int idx = chain_lookup_find(&lookup, name_ptr, name_len);
 */

#ifndef CHAIN_LOOKUP_H
#define CHAIN_LOOKUP_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

/* Power of 2 for fast modulo via bitmask */
#define CHAIN_LOOKUP_SIZE 512

typedef struct {
    char *name;      /**< Chain name (pointer into cif->names, not owned) */
    int index;       /**< Chain index, or -1 if empty */
} ChainLookupEntry;

typedef struct {
    ChainLookupEntry entries[CHAIN_LOOKUP_SIZE];
    int chain_count;
    bool initialized;
} ChainLookup;


/**
 * @brief FNV-1a hash for chain name.
 *
 * Standard FNV-1a 32-bit hash, good distribution for short strings.
 */
static inline uint32_t _chain_lookup_hash(const char *name, size_t len) {
    uint32_t hash = 2166136261u;  /* FNV offset basis */
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint8_t)name[i];
        hash *= 16777619u;  /* FNV prime */
    }
    return hash;
}


/**
 * @brief Initialize/clear a ChainLookup structure.
 *
 * Must be called before chain_lookup_build() or reusing a lookup.
 *
 * @param lookup Lookup structure to initialize
 */
static inline void chain_lookup_init(ChainLookup *lookup) {
    for (int i = 0; i < CHAIN_LOOKUP_SIZE; i++) {
        lookup->entries[i].name = NULL;
        lookup->entries[i].index = -1;
    }
    lookup->chain_count = 0;
    lookup->initialized = true;
}


/**
 * @brief Build chain lookup from names array.
 *
 * Populates the hash table with all chain names. Names are not copied;
 * the lookup stores pointers into the original names array.
 *
 * @param lookup Lookup structure (will be initialized if not already)
 * @param names Array of chain name strings
 * @param chain_count Number of chains
 */
static inline void chain_lookup_build(ChainLookup *lookup, char **names,
                                       int chain_count) {
    if (!lookup->initialized) {
        chain_lookup_init(lookup);
    }

    lookup->chain_count = chain_count;
    uint32_t mask = CHAIN_LOOKUP_SIZE - 1;

    for (int i = 0; i < chain_count; i++) {
        if (names[i] == NULL) continue;

        size_t len = strlen(names[i]);
        uint32_t h = _chain_lookup_hash(names[i], len);

        /* Linear probing to find empty slot */
        for (int j = 0; j < CHAIN_LOOKUP_SIZE; j++) {
            uint32_t idx = (h + (uint32_t)j) & mask;
            if (lookup->entries[idx].index < 0) {
                lookup->entries[idx].name = names[i];
                lookup->entries[idx].index = i;
                break;
            }
        }
    }
}


/**
 * @brief Look up chain index by name pointer and length.
 *
 * O(1) average case lookup using FNV-1a hash with linear probing.
 *
 * @param lookup Lookup structure (must be built)
 * @param name Pointer to chain name (not necessarily null-terminated)
 * @param len Length of chain name
 * @return Chain index (0-based), or -1 if not found
 */
static inline int chain_lookup_find(const ChainLookup *lookup,
                                     const char *name, size_t len) {
    if (lookup == NULL || name == NULL || !lookup->initialized) return -1;

    uint32_t h = _chain_lookup_hash(name, len);
    uint32_t mask = CHAIN_LOOKUP_SIZE - 1;

    for (int i = 0; i < CHAIN_LOOKUP_SIZE; i++) {
        uint32_t idx = (h + (uint32_t)i) & mask;
        const ChainLookupEntry *entry = &lookup->entries[idx];

        if (entry->index < 0) return -1;  /* Empty slot - not found */

        if (strlen(entry->name) == len &&
            strncmp(entry->name, name, len) == 0) {
            return entry->index;
        }
    }
    return -1;
}


/**
 * @brief Look up chain index by null-terminated name.
 *
 * Convenience wrapper for chain_lookup_find().
 *
 * @param lookup Lookup structure (must be built)
 * @param name Null-terminated chain name string
 * @return Chain index (0-based), or -1 if not found
 */
static inline int chain_lookup_find_str(const ChainLookup *lookup,
                                         const char *name) {
    if (name == NULL) return -1;
    return chain_lookup_find(lookup, name, strlen(name));
}


#endif /* CHAIN_LOOKUP_H */
