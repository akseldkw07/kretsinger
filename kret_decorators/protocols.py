from typing import Protocol, runtime_checkable


@runtime_checkable
class DefinesPostInit(Protocol):
    """Protocol for classes that define a post-init hook method."""

    def __post_init__(self) -> None:
        """Post-initialization hook called after __init__."""
        ...
