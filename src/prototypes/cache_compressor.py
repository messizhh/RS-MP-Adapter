class CacheCompressor:
    """Reserved cache compressor interface."""

    def compress(self, *args, **kwargs):
        raise NotImplementedError("Cache compression is out of scope for Phase 1A.")
