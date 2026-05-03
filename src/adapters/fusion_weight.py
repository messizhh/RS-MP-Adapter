class FusionWeight:
    """Reserved fusion-weight interface for RS-CPC."""

    def __init__(self, mode: str = "fixed_alpha") -> None:
        self.mode = mode
