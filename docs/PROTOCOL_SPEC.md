# Online TTA protocol specification (v1.0)

## Model of computation

A stream is an ordered list of frames x_1 … x_N grouped into sequences. A method holds a state θ (trainable
parameters, teacher, optimizer moments, memories such as trackers, running statistics, RNG). For every frame t:

1. if t starts a sequence: `on_sequence_start` (state components change only as declared in `boundary_policy`);
2. **predict**: ŷ_t = f_{θ_{t−1}}(x_t) — must not change θ;
3. **record** ŷ_t (exactly once, for this frame only);
4. **adapt**: θ_t = U(θ_{t−1}, x_t) — may use frame t and earlier frames, never labels.

Ground truth is only joined with the recorded predictions by the evaluator after the stream has ended.

## Checks

| Id | Name | Rule | Evidence in the trace |
|---|---|---|---|
| P1 | predict-before-update | ŷ_t is produced from θ_{t−1} | `predict` precedes `adapt`, and the state fingerprint after `predict` equals the one before it |
| P2 | state continuity | state at the start of frame t equals state at the end of frame t−1; at a sequence boundary a component may change only if its policy is `reset` | state fingerprints / parameter hashes before and after every frame |
| P3 | boundary policy | boundary handling happens exactly at sequence starts | one `sequence_start` event at every sequence boundary and none elsewhere |
| P4 | stream integrity | processed frames equal the declared stream (no duplicates, drops, reordering) | per-frame image ids vs. the declared manifest |
| P5 | single emission | exactly one emission per frame, only for that frame's image id | `emit` events and emitted image ids |
| P6 | label isolation | methods never receive labels | by construction: `FrameView` carries pixels and stream position only; canary test: corrupting all ground truth leaves predictions unchanged |

## Boundary policy of the shipped Detectron2 profiles

`source_only`, `mean_teacher`, `tent` and `whw` adapt continually over the whole stream: every state component
(student and teacher adapters, optimizer moments, WHW running statistics, RNG states) is **kept** at sequence
boundaries. A method with per-sequence state (e.g. a tracker) declares that component `reset`; the audit then accepts
a change of that component, and only at a sequence boundary.

## Trace contents and comparison

Per frame (`trace.jsonl`): image id, sequence, boundary flag, event sequence, emitted boxes / scores / classes,
decision (`update` / `skip_gate` / `skip_when` / `no_adaptation`) and reason, losses, method-specific update details
(e.g. the number of pseudo-labels), and SHA-256 fingerprints of the state at four points (before the boundary hook,
before `predict`, after `predict`, after `adapt`): student and teacher adapters, optimizer state, extra method state,
torch and NumPy RNG states.

`uav_shiftlab.reproducibility.trace_compare` compares two traces of the same stream field by field. On CPU (fp32,
fixed thread count) two runs of the same code are bitwise identical, so a code change that is meant to preserve
behaviour must reproduce every field (`tools/trace_capture.py --ref`), on the same CPU type: floating-point results
can differ between CPU models. On GPU, kernels are not bitwise deterministic; two runs are compared by their
decisions, emitted boxes and the final metric (`uavsl compare`), with the run-to-run variation of the same code (A/A)
as the reference scale.
