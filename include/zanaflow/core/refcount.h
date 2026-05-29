#ifndef REFCOUNT_H
#define REFCOUNT_H

typedef struct {
    int count;
} RefCount;

static inline void ref_init(RefCount *r) { r->count = 1; }
static inline void ref_retain(RefCount *r) { if (r) r->count++; }
static inline int ref_release(RefCount *r) { return (r && --r->count == 0); }

#endif
