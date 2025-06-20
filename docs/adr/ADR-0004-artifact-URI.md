# ADR-0004: Unified `artifact://` URI scheme

Date: 2025-06-20
Status: Proposed

## Context
CI workflows, nightly jobs and live serve instances currently hard-code disparate paths for model checkpoints, Parquet partitions and HTML reports. This makes local ↔️ cloud portability brittle and complicates cache invalidation.

## Decision
Introduce a lightweight URI abstraction:

```
artifact://<abs-path>                # local filesystem (default)
artifact+s3://bucket/key             # Amazon S3 (read-only v0)
artifact+gs://bucket/key             # Google Cloud Storage (todo)
```

`wnba.python.utils.artifact.open(uri, mode)` returns a file-like object.  The helper is intentionally minimal – no dependency on fsspec or smart-open – to keep the serve image slim.

## Consequences
1. Call-sites migrate gradually (search-replace of hard paths).
2. Future storage backends (R2, Azure) slot in behind the helper.
3. Local paths remain trivially usable; devs aren't forced onto S3.
4. Security: write support is local-only for now; cloud writes raise `UnsupportedSchemeError` until IAM story solidifies.

## Open Questions
• Should we vend a streaming iterator for large binary models? (follow-up)  
• How do we expose browse links in the HTML report?  
• Versioned URIs (`artifact://models/best@{sha}`) – punt to Edge-34.