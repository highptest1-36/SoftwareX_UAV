# SPDX-License-Identifier: Apache-2.0
"""Plugin interface for online TTA methods.

Contract (checked by uav_shiftlab.protocols):
  * predict(view) uses the state theta_{t-1} only and must not change it;
  * adapt(view, cache) may change the state using frame t (and the past) only;
  * state components keep or reset at sequence boundaries exactly as declared in `boundary_policy`.
"""
from typing import Any, Dict, Tuple

from .types import Detections, FrameView, UpdateEvent

KEEP, RESET = "keep", "reset"


class OnlineMethod:
    name = "base"
    # state component -> "keep" | "reset" at a sequence boundary (inside one stream)
    boundary_policy: Dict[str, str] = {}

    def on_stream_start(self) -> None:
        """Called once before the first frame."""

    def on_sequence_start(self, seq_id: str) -> None:
        """Called before the first frame of every sequence (including the first one)."""

    def predict(self, view: FrameView) -> Tuple[Detections, Any]:
        """Online prediction for frame t with theta_{t-1}. Returns (detections, cache passed to adapt)."""
        raise NotImplementedError

    def adapt(self, view: FrameView, cache: Any) -> UpdateEvent:
        """theta_t = U(theta_{t-1}, x_t). Default: no adaptation."""
        return UpdateEvent("no_adaptation")

    def state_dict(self) -> Dict[str, Any]:
        """Complete adaptation state (parameters, optimizer, memories, RNG) for mid-stream checkpoints."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoint/resume")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoint/resume")

    def state_fingerprint(self) -> Dict[str, str]:
        """Component -> hash of the current state (used for continuity and boundary checks)."""
        return {}
