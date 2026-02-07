Atlas Memory Contract

1. Single contiguous allocation per Workspace.
2. Allocation is 128-byte aligned.
3. 16KB page pre-touch on construction.
4. Layout derived from BM, BN, BK, MR, NR.
5. Regions:
   - A pack region
   - B pack region
   - Accumulator region
6. No dynamic resizing.
7. No locking.
8. One workspace per thread.
9. Reset zeroes only accumulator region.
10. Overflow is fatal.

