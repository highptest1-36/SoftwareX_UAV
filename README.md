# UAV-ShiftLab

**An auditable and resumable framework for online test-time adaptation of UAV video object detectors.**

License: [Apache-2.0](LICENSE) (see also [NOTICE](NOTICE)).

Online TTA results are easy to get subtly wrong: a model may be updated on a frame before its prediction for that
frame is recorded, state may be reset (or not) at sequence boundaries without anyone noticing, frames may be
processed twice after a crash, and two methods may be compared on different stream orders. UAV-ShiftLab runs
online TTA methods frame by frame, records a complete protocol trace, audits it, and makes long GPU runs
interruptible and resumable without changing their result.

## Components

| Component | Path |
|---|---|
| Label-free stream manifests (original / reverse / permuted sequence orders, fingerprints) | `src/uav_shiftlab/core/stream.py` |
| Plugin interface (`predict` / `adapt` / declared boundary policy / state fingerprints) | `src/uav_shiftlab/core/method.py` |
| Online runner (predict → record → adapt) with protocol traces and mid-stream checkpoint/resume | `src/uav_shiftlab/core/runner.py` |
| Protocol audit P1–P5 (predict-before-update, state continuity, boundary policy, stream integrity, single emission) | `src/uav_shiftlab/protocols/audit.py` |
| Deliberate protocol violations for impact studies (adapt-before-predict, silent per-sequence reset) | `src/uav_shiftlab/protocols/violations.py` |
| Frame-by-frame trace comparison with tolerance profiles | `src/uav_shiftlab/reproducibility/trace_compare.py` |
| Run manifests (environment, code version, SHA-256 of every input) | `src/uav_shiftlab/reproducibility/manifest.py` |
| COCO-style evaluation for UAV video (class-agnostic and 3-class, tiny < 16 px, ignore regions, per sequence) | `src/uav_shiftlab/evaluation/coco_eval.py` |
| Diagnostics: windowed AP (drift), stream-order sensitivity, pseudo-label audit; fixed-reference forgetting | `src/uav_shiftlab/diagnostics/`, `tools/forgetting_eval.py` |
| Mock backend: synthetic day/night streams, toy detector and methods (NumPy only, for tests and the demo) | `src/uav_shiftlab/backends/mock/` |
| Detectron2 backend: Faster R-CNN R50-FPN with trainable FPN adapters; Source only, Mean Teacher, Tent, WHW | `src/uav_shiftlab/backends/detectron2/` |
| Command line `uavsl` (demo, run, audit, eval, compare, validate) | `src/uav_shiftlab/cli.py` |

## Quick start (CPU, no GPU, no dataset)

```bash
pip install -e ".[dev]"
python -m pytest -q            # engine, protocol audit (incl. deliberate violations), trace comparison, resume
uavsl demo --out outputs/demo  # synthetic day -> night stream, two toy methods, audit, ~10 s
```

```python
from uav_shiftlab.backends.mock.synthetic import make_dataset
from uav_shiftlab.backends.mock.methods import SourceOnly, AdaptiveThreshold
from uav_shiftlab.core.stream import StreamManifest
from uav_shiftlab.core.runner import OnlineRunner
from uav_shiftlab.protocols.audit import audit_trace
from uav_shiftlab.evaluation.coco_eval import evaluate

coco, images = make_dataset()                      # day sequence followed by two night sequences
stream = StreamManifest.from_coco(coco, image_root="")
for method in (SourceOnly(), AdaptiveThreshold()):
    runner = OnlineRunner(method, stream, load=lambda p: images[p])
    trace = runner.run()
    ap = evaluate(coco, runner.coco_detections(), modes=("agnostic",))["agnostic"]["AP50"]
    print(method.name, "AP50", ap, "protocol OK:", audit_trace(trace)["pass"])
```

The synthetic streams demonstrate the software (protocol, state handling, evaluation); they make no claim about
detection performance on real aerial video.

## Writing a method

A method implements `predict(view) -> (Detections, cache)` (must not change its state), `adapt(view, cache) ->
UpdateEvent`, `state_fingerprint()` (hashes of every state component), optionally `state_dict()` /
`load_state_dict()` for checkpoint/resume, and declares for every state component whether it is kept or reset at a
sequence boundary (`boundary_policy`). See `src/uav_shiftlab/backends/mock/methods.py` for small complete examples and
`src/uav_shiftlab/backends/detectron2/plugins.py` for a real detector.

## Detectron2 methods on UAVDT (optional backend)

```bash
uavsl run --method mean_teacher --coco /path/coco/test_night.json --images /path/UAV-benchmark-M \
          --assets-root /path/assets --out outputs/mt_s0 --seed 0 --checkpoint-every 250 --allow-test
uavsl audit outputs/mt_s0                    # P1-P5
uavsl compare outputs/mt_s0 outputs/mt_s0b   # frame-by-frame divergence of two runs (A/A, A/B)
uavsl validate outputs/mt_s0                 # inputs unchanged since the run? profile files intact?
```

Profiles (`src/uav_shiftlab/backends/detectron2/profiles/uavdt/`, SHA-256 in `PROFILE_SHA256.json`):
`source_only`, `mean_teacher` (fixed confidence threshold 0.7, EMA teacher), `tent`, `whw`. All share the same frozen
detector, the same residual FPN adapters (the only trainable parameters) and the same stream protocol, so only the
adaptation objective differs.

A run directory holds `manifest.json`, `config.resolved.yaml`, `trace.jsonl`, `predictions.jsonl`, `checkpoint.pkl`,
`metrics.json`, `audit.json` (and `snapshots/` with `--snapshot-every`). Re-running the same command after an
interruption resumes from the last checkpoint and gives the same result as an uninterrupted run. Test splits require
`--allow-test`: tune on development splits only.

Installation of PyTorch/Detectron2 and the expected data/asset layout: [docs/INSTALL.md](docs/INSTALL.md).

## Results and reproducibility

[docs/RESULTS.md](docs/RESULTS.md) (generated from `results/runs_summary.json` by `tools/results_report.py`) reports
the agreement of the engine with the original research code on an A100, the main runs (3 seeds), forgetting, the
protocol-violation impact study and stream-order sensitivity. `experiments/run_experiments.sh` reruns sections 2–4.

On CPU, the Detectron2 methods are bitwise deterministic for a given thread count: `tools/trace_capture.py` records
complete traces on short development-split fixtures and compares a new code version with them field by field;
`tools/run_resume_check.sh` interrupts runs in one process and resumes them in another, and requires the resumed
trace to equal the uninterrupted one. The protocol and the trace format are specified in
[docs/PROTOCOL_SPEC.md](docs/PROTOCOL_SPEC.md).

## Data

The experiments use **UAVDT** (Du et al., ECCV 2018), which its authors provide for research purposes only. It is
**not** redistributed here. Download from the official page, https://sites.google.com/view/grli-uavdt/
(mirror listing the same original files: https://github.com/dataset-ninja/uavdt/blob/main/DOWNLOAD.md):

- *UAVDT-Benchmark-M* (images, `UAV-benchmark-M/`)
- *DET/MOT toolkit* (ground truth, `UAV-benchmark-MOTD_v1.0/GT/`)
- *Attributes* (`M_attr/`)

then build the splits and COCO files with `tools/prepare_uavdt.py` ([docs/INSTALL.md](docs/INSTALL.md)). The source
detector checkpoint and the WHW statistics are published separately on Zenodo, https://doi.org/10.5281/zenodo.23598442 (see docs/INSTALL.md). If you use the
data, cite the UAVDT paper:

```bibtex
@inproceedings{du2018uavdt,
  title     = {The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking},
  author    = {Du, Dawei and Qi, Yuankai and Yu, Hongyang and Yang, Yifan and Duan, Kaiwen and Li, Guorong and
               Zhang, Weigang and Huang, Qingming and Tian, Qi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2018}
}
```

## Repository layout

```
src/uav_shiftlab/   core · protocols · evaluation · diagnostics · reproducibility · datasets · backends/{mock,detectron2}
tests/              CPU tests (Detectron2 tests are skipped when it is not installed)
tools/              UAVDT preparation, source training/statistics, trace capture/compare, resume check,
                    forgetting evaluation, run summaries, results report
experiments/        GPU runbook of the reported experiments
results/            summaries of the reported runs (input of docs/RESULTS.md)
docs/               installation, protocol specification, results
```

## Citation

If you use UAV-ShiftLab, please cite the software (metadata in [CITATION.cff](CITATION.cff); GitHub's
"Cite this repository" button gives APA and BibTeX):

```bibtex
@software{uavshiftlab2026,
  title   = {UAV-ShiftLab: An auditable and resumable framework for online test-time adaptation of UAV video object detectors},
  author  = {Ha, Cao-Phuc and Nguyen Vu Minh, Anh and Nguyen Quoc, Huy and Nguyen Van, Bay and Duong Thi Hong, Ha and Ho Huong, Thien and Lam Tran Tuan, Dzi and Le, Phu-Nguyen},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/highptest1-36/SoftwareX_UAV},
  license = {Apache-2.0}
}
```

A journal article describing the software is in preparation; its reference will be added here once published.

## Contact

Questions and bug reports: GitHub issues, or phuchc36@gmail.com and nguyenlp@vaa.edu.vn.
