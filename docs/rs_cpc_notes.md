# RS-CPC Notes

The intended main method is a compact multi-prototype cache adapter for remote sensing few-shot scene classification.

Prompt learning, optimal transport alignment, and unrelated adapter methods are not part of the main Phase 1A implementation.

Phase 1E adds a training-free RS-CPC skeleton over cached features. It supports compact prototype counts `M = 1, 2, 4, 8`, reports cache entries as `C x M`, and reports compression ratio against the sample-level support cache.

Prototype initialization currently supports mean, random group mean, medoid, and a dependency-free kmeans fallback to deterministic random grouping. Fine-tuned RS-CPC is not implemented in this phase.

Phase 1F validates the training-free RS-CPC runner inside the local fake pipeline. This uses synthetic data and fake features, reports non-paper smoke metadata, and only checks implementation wiring and cache-size accounting. It is not evidence of real remote-sensing accuracy.

Future real RS-CPC sweeps must run on the remote server from verified feature caches and splits. Server templates are placeholders only and must be edited with real dataset, feature, weight, and output roots before manual execution.
