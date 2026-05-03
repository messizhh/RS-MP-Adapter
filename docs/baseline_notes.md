# Baseline Notes

Phase 1E implements feature-cache-based, training-free baseline skeletons for local fake-cache validation:

- zero-shot cached evaluation
- linear probe skeleton with a CPU nearest-centroid fallback
- Tip-Adapter training-free cache logits
- Proto-Adapter training-free prototype logits

Fine-tuned variants are not implemented yet. `Tip-Adapter-F` and `Proto-Adapter-F` must not be treated as completed baselines.

Local smoke outputs are fake/non-paper validation artifacts. Real baseline results require verified real splits and real server-generated feature caches later.
