# Shopping Sessions — Rust pipeline

Single multi-threaded Rust binary implementing `rules.md` exactly.

Design (v2, speed):
- IDs (≤15 bytes) packed into 128-bit keys whose order equals byte order; longer IDs fall back to a hash + side table
- AVX2 CSV scanner (64-byte blocks → newline/comma/quote bitmasks), branch-free field closing
- Timestamps parsed by hand with a per-thread date cache
- Links: hash interning + union-find; person = smallest packed key
- Events partitioned by `event_id` hash → in-cache open-addressing dedup (smallest `ingest_ts`, before validation)
- Partition join: survivors and clicks regrouped by user-id hash, resolved to people with small per-bucket tables, regrouped by person
- Per person bucket: counting sort + small sorts, sessionize, 7-day sliding window for credit and assists
- mimalloc with huge pages; output written as 8 parallel Parquet parts; exit without freeing

Verified: 36/36 fixtures, exact match with `dev-answer/`, and identical output to the v1 reference on 6 extra seeds (incl. a Feb-29 window) and a 20M-event set.

Build & run:

    docker build -t shopping-sessions .
    docker run --rm --network none --user 1000:1000 --read-only \
      -v $PWD/dev:/data:ro -v $PWD/out:/output shopping-sessions

Without Docker: `cargo build --release && DATA_DIR=dev OUTPUT_DIR=out VERBOSE=1 target/release/sessions`
