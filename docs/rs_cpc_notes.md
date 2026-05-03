# RS-CPC Notes

The intended main method is a compact multi-prototype cache adapter for remote sensing few-shot scene classification.

Prompt learning, optimal transport alignment, and unrelated adapter methods are not part of the main Phase 1A implementation.

Phase 1E adds a training-free RS-CPC skeleton over cached features. It supports compact prototype counts `M = 1, 2, 4, 8`, reports cache entries as `C x M`, and reports compression ratio against the sample-level support cache.

Prototype initialization currently supports mean, random group mean, medoid, and a dependency-free kmeans fallback to deterministic random grouping. Fine-tuned RS-CPC is not implemented in this phase.
