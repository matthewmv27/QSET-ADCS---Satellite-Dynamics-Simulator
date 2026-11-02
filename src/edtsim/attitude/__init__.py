"""Standalone rigid-bus attitude dynamics (no orbit or flexible tether)."""
from .rigid_body import AttitudeHistory, propagate_attitude, rigid_body_rhs

__all__ = ["AttitudeHistory", "propagate_attitude", "rigid_body_rhs"]
