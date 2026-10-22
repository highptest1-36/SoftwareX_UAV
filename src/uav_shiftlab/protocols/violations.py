# SPDX-License-Identifier: Apache-2.0
"""Deliberate, realistic protocol violations - used to quantify how much each one changes the reported metrics
(Protocol-Violation Impact Study) and to demonstrate that the audit detects them. Never use them for results.

  adapt_before_predict   theta_t = U(theta_{t-1}, x_t) first, then y_t = f_{theta_t}(x_t)   (test-time leakage of x_t)
  reset_each_sequence    the whole adaptation state is re-initialised at every sequence start while the method
                         declares that it keeps it (silent per-sequence reset)
"""
from ..core.method import OnlineMethod
from ..core.runner import OnlineRunner


class AdaptBeforePredictRunner(OnlineRunner):
    def _process(self, view, ev, rec):
        upd = self.method.adapt(view, self.method.predict(view)[1])   # update on x_t ...
        ev.append("adapt")
        dets, _ = self.method.predict(view)                           # ... then predict with theta_t
        ev.append("predict")
        rec["state_after_predict"] = self.method.state_fingerprint()
        ev.append("emit")
        return dets, upd


class ResetEachSequence(OnlineMethod):
    """Proxy: delegates everything to `inner`, but restores its initial state at every sequence start."""

    def __init__(self, inner):
        self.inner, self.name = inner, inner.name + "+reset_each_sequence"
        self.boundary_policy = dict(inner.boundary_policy)   # still declares "keep" -> the audit must flag it
        self._init = None

    def on_stream_start(self):
        self.inner.on_stream_start()
        self._init = self.inner.state_dict()

    def on_sequence_start(self, seq_id):
        if self._init is not None:
            self.inner.load_state_dict(self._init)
        self.inner.on_sequence_start(seq_id)

    def predict(self, view):
        return self.inner.predict(view)

    def adapt(self, view, cache):
        return self.inner.adapt(view, cache)

    def state_dict(self):
        return {"inner": self.inner.state_dict(), "init": self._init}

    def load_state_dict(self, sd):
        self.inner.load_state_dict(sd["inner"])
        self._init = sd["init"]

    def state_fingerprint(self):
        return self.inner.state_fingerprint()

    def __getattr__(self, k):            # e.g. `.r` of the Detectron2 plugins
        return getattr(self.__dict__["inner"], k)


VIOLATIONS = ("none", "adapt_before_predict", "reset_each_sequence")


def apply(violation, method, runner_cls=OnlineRunner):
    if violation == "none":
        return method, runner_cls
    if violation == "adapt_before_predict":
        return method, AdaptBeforePredictRunner
    if violation == "reset_each_sequence":
        return ResetEachSequence(method), runner_cls
    raise ValueError(violation)
