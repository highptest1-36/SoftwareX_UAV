# Installation

## Core (CPU, any OS)

```bash
pip install -e ".[dev]"          # NumPy, SciPy, pycocotools, PyYAML, pytest
pip install faster-coco-eval     # optional: ~30x faster evaluation, numerically identical
python -m pytest -q
```

## Detectron2 backend (optional)

Detectron2 has no official PyPI wheels. The reference results were produced with Python 3.13, PyTorch 2.11 and
Detectron2 built from source at commit `1e3e13bbf607b54f62205c4c33922521822fb298`:

```bash
pip install "setuptools<81" fvcore iopath opencv-python
git clone https://github.com/facebookresearch/detectron2.git && cd detectron2
git checkout 1e3e13bbf607b54f62205c4c33922521822fb298
pip wheel --no-build-isolation --no-deps . -w wheels && pip install --no-deps wheels/detectron2-*.whl
```

On GPU builds set `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST` before building.

## Data and assets for the Detectron2 methods

`uavsl run` needs three inputs besides the code:

1. **UAVDT images** (`--images`): the `UAV-benchmark-M/<sequence>/img<frame>.jpg` tree from the dataset authors.
2. **A COCO json per split** (`--coco`): `images` with `id`, `file_name` (relative to `--images`, e.g.
   `M0203/img000001.jpg`), `width`, `height`, `video` (sequence id) and `frame_id` (1-based); `annotations` in pixel
   coordinates with category ids 1 car, 2 truck, 3 bus; UAVDT ignore regions as `iscrowd: 1` annotations (detections
   inside them are neither true nor false positives). The online run reads only `images`; labels are used by the
   evaluator after the stream has ended.
3. **An assets root** (`--assets-root`) with `checkpoints/source_day_r50_1080/model_final.pth` (the source detector,
   `profiles/uavdt/detector.yaml`) and, for WHW, `checkpoints/source_day_r50_1080/source_stats.pt` (source feature
   statistics).

How to obtain them:

```bash
pip install -e ".[data]"
python3 tools/prepare_uavdt.py --uavdt-root /path/UAVDT --out /path/assets/data   # splits + COCO json (~2 min)
```

`--uavdt-root` contains the official `UAV-benchmark-M/`, `UAV-benchmark-MOTD_v1.0/GT/` and `M_attr/` folders
(`--images`, `--gt-dir`, `--attr-dir` override them; the official archives may extract into nested folders such as
`M_attr/M_attr/` — point the options at the folders that contain the `.txt` files). The source checkpoint and the WHW statistics used for
docs/RESULTS.md are published on Zenodo, https://doi.org/10.5281/zenodo.23598442; place them in
`<assets>/checkpoints/source_day_r50_1080/`. To recreate them instead (GPU):

```bash
python3 tools/train_source.py --assets-root /path/assets --images /path/UAVDT/UAV-benchmark-M        # ~10k iterations
python3 tools/source_stats.py --assets-root /path/assets --images /path/UAVDT/UAV-benchmark-M        # 2000 images
```

A retrained detector differs from the published checkpoint (GPU non-determinism), so results will differ slightly.
Recomputing `source_stats.pt` with the published checkpoint reproduces the image selection and object counts exactly,
but the feature statistics depend on the numerical precision: on CPU (fp32) the class-wise object-feature means differ
from the published file (GPU, bf16 autocast) by up to about 5 % (relative L2), the image-level statistics by < 0.3 %.
Use the published file to reproduce the WHW results.

Sequence splits used in docs/RESULTS.md (UAVDT-M sequence ids):

| split | sequences | use |
|---|---|---|
| source_train | M0101 M0201 M0207 M0401 M0402 M0603 M0604 M0605 M0702 M0703 M1201 M1304 M1305 M1306 | training of the source detector (daylight) |
| source_val | M0210 M0301 M1202 | validation of the source detector |
| dev_night | M0202 M0204 M0206 M0901 M0902 M1002 M1003 M1102 | hyper-parameter selection, CPU fixtures |
| test_night | M0203 M0205 M1001 M1007 M1101 | online adaptation (reported) |
| test_day | M0208 M0209 M0403 M0601 M0602 M0606 M0801 M0802 M1301 M1302 M1303 M1401 | fixed-reference forgetting |
