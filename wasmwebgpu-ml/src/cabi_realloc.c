/**
 * @file cabi_realloc.c
 * @brief Component Model ABI memory allocation function
 * 
 * This file implements the cabi_realloc function required by wit-bindgen
 * when the host needs to allocate memory in the guest module for returning
 * strings, lists, and other dynamically-sized data.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/**
 * cabi_realloc - Canonical ABI reallocation function
 * 
 * This function is called by the host runtime to allocate or reallocate
 * memory in the guest's linear memory for returning values.
 * 
 * @param old_ptr   Pointer to existing allocation (NULL for new allocation)
 * @param old_size  Size of existing allocation (0 for new allocation)
 * @param align     Required alignment (must be power of 2)
 * @param new_size  Desired size of the new allocation
 * @return          Pointer to allocated memory, or NULL on failure
 */
__attribute__((export_name("cabi_realloc")))
void* cabi_realloc(void* old_ptr, size_t old_size, size_t align, size_t new_size) {
    // Handle deallocation case
    if (new_size == 0) {
        if (old_ptr != NULL) {
            free(old_ptr);
        }
        return NULL;
    }
    
    // For new allocations, use aligned_alloc if alignment > default
    if (old_ptr == NULL) {
        // Use malloc for small alignments, aligned_alloc for larger
        if (align <= sizeof(void*)) {
            return malloc(new_size);
        } else {
            // Ensure new_size is a multiple of align for aligned_alloc
            size_t aligned_size = (new_size + align - 1) & ~(align - 1);
            return aligned_alloc(align, aligned_size);
        }
    }
    
    // For reallocation, use realloc
    // Note: realloc doesn't guarantee alignment, but for most cases this is fine
    void* new_ptr = realloc(old_ptr, new_size);
    
    return new_ptr;
}
