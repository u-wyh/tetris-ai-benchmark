"""Public name for the no-Hold browser V1 heuristic adapter."""

from .v1 import V1Adapter as LegacyHeuristic

NAME = "legacy_heuristic"

__all__ = ["LegacyHeuristic", "NAME"]
