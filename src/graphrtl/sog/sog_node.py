"""Dataclass for a node in a simple operator graph (SOG)."""

from dataclasses import dataclass


@dataclass(slots=True)
class SOGNode:
    """Dataclass for a node in a simple operator graph (SOG)."""

    name: str
    type: str
    bit_width: int = 1
    fanout: int = 1
    toggle_rate: float = 0.5
    transition_time: float = 0.5
    delay: float = 0.0

    def __repr__(self) -> str:
        """Represent the SOGNode by its name."""
        return self.name
