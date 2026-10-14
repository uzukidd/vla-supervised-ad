# vla-supervised-ad

**English** | [简体中文](README.zh-CN.md)

Use local SAM3 for text-prompted segmentation of camera images from Sidewalk Semantic, nuScenes mini, and nuImages, and export class maps, visualizations, and evaluation metrics. nuScenes supports lidarseg projection evaluation and colored point clouds; nuImages is evaluated against image-mask ground truth.

Suggested reading order: [Environment setup](#environment-setup) → [Command reference](#command-reference) → [Project layout and data paths](#project-layout-and-data-paths). The following sections describe each dataset, point-cloud outputs, and evaluation conventions in detail.

## Environment setup

Run all commands below in **WSL Bash**, with the project at `/mnt/d/wsl_workplace/vla-supervised-ad`. The current distribution is Ubuntu-26.04, using the Conda environment **`vla-supervised`** with `cuda + bf16` inference by default. Actual inference has been verified on this machine's NVIDIA RTX 5070 Ti.

### Use the current environment

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# Activate manually to run Python or inspect dependencies.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla-supervised
python --version
nvidia-smi
```

The current environment already contains all runtime dependencies. For routine use, run the Bash entry points below directly. [run_vla.sh](scripts/run_vla.sh) selects `vla-supervised`, using `conda run -n vla-supervised` when it has not been manually activated; it does not use base or Windows Python. The current interpreter is `/home/administrator/miniconda3/envs/vla-supervised/bin/python`. If Conda is installed elsewhere, activate this environment first or set `CONDA_EXE=/path/to/conda/bin/conda`.

| Component | Currently installed version / path | Purpose |
|---|---|---|
| Python | 3.12.14 | Project interpreter |
| torch | 2.14.1+cu132 | GPU inference |
| torchvision | 0.29.1+cu132 | SAM3 image preprocessing |
| sam3 | 0.1.0, editable source at `/mnt/d/wsl_workplace/sam3` | Official SAM3 image model |
| NumPy | 2.5.3 | Masks, projection, fusion, and metrics |
| Pillow | 12.3.0 | Image I/O and 2D visualization |
| PyArrow | 26.0.0 | Reading Sidewalk Parquet files |
| pycocotools | 2.0.11 | Decoding nuImages RLE masks |
| matplotlib | 3.11.2 | Static point-cloud previews |
| einops | 0.8.2 | Internal SAM3 dependency |
| setuptools | 80.10.2 | `pkg_resources` used by the current SAM3 |

This table records the verified installation on the current machine; it is not an environment lock file for other machines. The current SAM3 package declares `numpy<2`, while the existing environment uses NumPy 2.5.3, so `pip check` reports this version mismatch. Actual inference and regression tests have passed for the project's modes. The new-environment instructions below install NumPy 1.x according to SAM3's declared requirement; the existing environment does not need to be modified to run these commands.

### Create an environment on another WSL installation

Run `conda env list` first. If `vla-supervised` already exists, use it directly; run `conda create` only when that environment is absent. These are installation instructions; the existing environment was not recreated or replaced for this update.

```bash
# Run in WSL with Miniconda installed; adjust the path to your installation.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env list

# Run only if vla-supervised does not already exist.
conda create -n vla-supervised python=3.12 -y
conda activate vla-supervised

# Choose matching torch/torchvision CUDA wheels supporting your NVIDIA GPU.
# cu128 and PyTorch 2.10.0 are the installation example in the local SAM3 README.
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install "numpy>=1.26,<2" "setuptools<81" einops matplotlib

# SAM3 source already exists on this machine; put the official source here on a new machine first.
# git clone https://github.com/facebookresearch/sam3.git /mnt/d/wsl_workplace/sam3
python -m pip install -e /mnt/d/wsl_workplace/sam3

cd /mnt/d/wsl_workplace/vla-supervised-ad
bash scripts/setup_sidewalk.sh
bash scripts/setup_nuimages.sh
```

[setup_sidewalk.sh](scripts/setup_sidewalk.sh) adds PyArrow, Pillow, and NumPy according to [requirements-sidewalk.txt](requirements-sidewalk.txt); [setup_nuimages.sh](scripts/setup_nuimages.sh) adds pycocotools, Pillow, and NumPy according to [requirements-nuimages.txt](requirements-nuimages.txt). Neither script installs SAM3, torch, or torchvision. nuScenes data reading and projection use NumPy/Pillow and do not require `nuscenes-devkit`. Point-cloud HTML uses offline WebGL and needs no web service or CDN. Static point-cloud PNGs require matplotlib.

The model uses the local official `models/sam3/sam3.pt`. The model directory and SAM3 source directory are `/mnt/d/model_zoo/sam3` and `/mnt/d/wsl_workplace/sam3`, respectively: the weights directory does not automatically provide the `sam3` Python package. Model construction explicitly sets `load_from_HF=False`, so segmentation does not need to download weights.

### Check the environment and data

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# Print the actual interpreter, GPU availability, key libraries, and SAM3 source path; do not load the model.
bash scripts/run_vla.sh - <<'PY'
import sys
import torch
import torchvision
import sam3
import numpy
import PIL
import pyarrow
import pycocotools.mask
import matplotlib
print("Python:", sys.executable)
print("torch / torchvision:", torch.__version__, torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("SAM3:", sam3.__file__)
print("NumPy / Pillow / PyArrow / matplotlib:", numpy.__version__, PIL.__version__, pyarrow.__version__, matplotlib.__version__)
PY

# Check the links. All source paths must exist; existing links do not need to be recreated.
ls -ld models/sam3 datasets/sidewalk-semantic datasets/nuscenes datasets/nuimages
readlink -f models/sam3 datasets/sidewalk-semantic datasets/nuscenes datasets/nuimages

# Read only data/annotations, without running SAM3.
bash scripts/segment_sidewalk.sh --inspect-data
bash scripts/segment_nuscenes.sh --inspect-data
bash scripts/segment_nuimages.sh --inspect-data
```

If CUDA is unavailable during the environment check, first verify `nvidia-smi` and torch's CUDA wheel in WSL. All segmentation entry points also support `--device cpu --precision fp32`; CPU execution is slower than GPU execution.

## Command reference

All commands below assume you are in the project root. Use a **new or empty output directory** for each segmentation run or point-cloud export. The examples in the table limit the number of samples and do not start full-dataset inference. Append `--help` to a segmentation/export entry point for its complete parameter list.

| Purpose | Ready-to-run command | Output / default behavior |
|---|---|---|
| Sidewalk dataset-class segmentation and evaluation | `bash scripts/segment_sidewalk.sh --max-images 5 --output-dir outputs/sidewalk_demo` | Predictions, GT comparison visualizations, and pixel metrics; train by default |
| Sidewalk custom TXT classes | `bash scripts/segment_custom.sh --classes-file config/classes.txt --max-images 5 --output-dir outputs/sidewalk_custom` | Predictions and visualizations, without GT metrics |
| nuScenes front camera, 16 classes | `bash scripts/segment_nuscenes.sh --max-images 5 --output-dir outputs/nuscenes_demo` | mini_val / CAM_FRONT by default; lidarseg projected-point observation evaluation |
| nuScenes original 32 classes | `bash scripts/segment_nuscenes.sh --taxonomy lidarseg32 --max-images 5 --output-dir outputs/nuscenes_raw32` | Preserves original lidarseg IDs |
| nuScenes six cameras + colored point cloud | `bash scripts/segment_nuscenes_6cams.sh --output-dir outputs/nuscenes_six_demo` | Defaults to 6 camera images = 1 LiDAR frame; automatic fusion, 3D export, and unique-point metrics |
| nuScenes six cameras over 10 frames | `bash scripts/segment_nuscenes_6cams.sh --max-images 60 --output-dir outputs/nuscenes_six_10frames` | 60 images = 10 frames, fused independently per frame |
| nuScenes custom TXT classes | `bash scripts/segment_nuscenes_custom.sh --classes-file config/classes.txt --max-images 5 --output-dir outputs/nuscenes_custom` | Image segmentation and visualization, without class GT metrics |
| nuScenes six cameras with custom TXT classes + point cloud | `bash scripts/segment_nuscenes_6cams.sh --mode custom --classes-file config/classes.txt --output-dir outputs/nuscenes_six_custom` | Predicted point cloud and coverage, without lidarseg semantic metrics |
| Color a point cloud using existing nuScenes segmentation results | `bash scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --output-dir outputs/nuscenes_demo_cloud` | No SAM3 rerun required; PLY, NPZ, HTML, PNG, and unique-point evaluation in dataset mode |
| Export only 1 point-cloud frame from existing results | `bash scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --max-frames 1 --output-dir outputs/nuscenes_first_cloud` | Count is in LiDAR keyframes |
| nuImages class segmentation, GT comparison, and evaluation | `bash scripts/segment_nuimages.sh --max-images 5 --output-dir outputs/nuimages_demo` | v1.0-mini by default, 24 non-ego classes; six-panel comparison and pixel metrics |
| nuImages front-view filtering only | `bash scripts/segment_nuimages.sh --cameras CAM_FRONT --max-images 5 --output-dir outputs/nuimages_front` | Selects only annotated key-camera images |
| nuImages prompting and evaluation including ego | `bash scripts/segment_nuimages.sh --include-ego --max-images 5 --output-dir outputs/nuimages_with_ego` | Includes ID31, for 25 prompted classes |
| nuImages custom TXT classes | `bash scripts/segment_nuimages_custom.sh --classes-file config/classes.txt --max-images 5 --output-dir outputs/nuimages_custom` | Image segmentation and visualization, without GT metrics |
| Create/check the nuImages symlink | `bash scripts/link_nuimages.sh` | Points to `/mnt/d/datasets/nuimages`; does not overwrite an existing different target |
| Add Sidewalk data-reading dependencies | `bash scripts/setup_sidewalk.sh` | Always installs into vla-supervised |
| Add nuImages mask-decoding dependencies | `bash scripts/setup_nuimages.sh` | Always installs into vla-supervised |
| All regression tests | `bash scripts/test_sidewalk.sh` | Despite the sidewalk name, runs all tests, including point-cloud and nuImages tests |
| nuScenes data/export pipeline tests | `bash scripts/test_nuscenes.sh` | Filters by `*nuscenes*.py`; does not separately include viewer/fusion files |
| nuImages data/pipeline tests | `bash scripts/test_nuimages.sh` | Filters by `*nuimages*.py` |
| Execute Python commands in the designated environment | `bash scripts/run_vla.sh -m pip list` | General environment entry point; accepts any Python arguments |

### Counts, parameters, and viewing results

| Parameter / default | Sidewalk | General nuScenes entry point | Six-camera nuScenes entry point | nuImages |
|---|---|---|---|---|
| Default number processed | All remaining images | 10 camera images | 6 camera images = 1 frame | 10 annotated images |
| Unit of `--max-images N` | Images | Camera images | Camera images; multiples of 6 recommended | Annotated images after filtering |
| `--max-images 0` | All remaining images | All selected views | All selected six-camera views | All remaining keyframes |
| Data selection | `--split train/validation/test` | `--split all/mini_train/mini_val` | Same as nuScenes | `--version v1.0-mini` / `v1.0-train` / `v1.0-val` / `v1.0-test` |
| Class mode | Dataset / custom | `lidarseg16/lidarseg32` / custom | Same as nuScenes | Dataset / custom; can include ego |

Segmentation entry points for all three datasets support `--start-index`, `--classes-file`, `--score-threshold`, `--mask-threshold`, `--device`, `--precision`, `--alpha`, `--vis-every`, `--no-vis`, `--inspect-data`, and `--output-dir`. Both thresholds default to 0.5, and the default execution mode is `cuda/bf16`. Override model and data paths with `--checkpoint` and `--dataset-dir`. The selected split/version must actually exist; locally, Sidewalk has only train, and nuScenes and nuImages have only mini.

nuScenes uses `--cameras`, `--scenes`, `--min-depth`, and `--export-pointcloud`. In nuImages, `--cameras` filters annotated key cameras; it does not mean that each sample has six annotations. The standalone point-cloud export entry point supports `--max-frames`, `--html-max-points`, `--no-html`, and `--no-preview`. `--html-max-points` affects display sampling only; metrics and PLY/NPZ retain all points.

- 2D comparison images: `visualizations/` in each segmentation output; raw class maps are in `predictions/`.
- Image metrics / camera projected-point observation metrics: `metrics.json` and `metrics.csv` at the segmentation output root.
- Six-camera point clouds: `pointcloud_results/index.html` in the segmentation output, which can be opened directly in a Windows browser; point-cloud PLY/NPZ/PNG files are in its `pointclouds/` directory.
- Fused unique-point metrics: `evaluation/metrics.json` in the point-cloud output, reporting covered_points, full_scan, and coverage separately. Their scope differs from the projected-observation metrics at the segmentation output root.
- Custom TXT mode does not output semantic GT evaluation; custom nuScenes point clouds still report geometric coverage.

Class mappings, output fields, and evaluation details for each mode are described in the following sections.

## Project layout and data paths

Only **WSL + Conda `vla-supervised`** is supported. The launcher scripts select this environment rather than base or Windows Python. Python implementations are in `python/`, `scripts/` contains only Shell launcher scripts, and class configurations are in `config/`.

```text
python/
  sam3_sidewalk.py          # Data reading, SAM3 inference, instance merging, and visualization
  sam3_nuscenes.py          # nuScenes camera segmentation and lidarseg projected-point evaluation
  sam3_nuimages.py          # nuImages image segmentation, GT comparison, and pixel metrics
  nuimages_data.py          # nuImages keyframes and base64 RLE mask decoding
  nuscenes_data.py          # mini metadata, class mappings, and coordinate projection across timestamps
  nuscenes_pointcloud.py    # Camera pixel-class assignment, point-cloud export, and unique-point evaluation
  pointcloud_semantics.py   # Cross-camera fusion, PLY, coverage, and two evaluation scopes
  pointcloud_viewer.py      # Offline interactive 3D HTML and static previews
  segmentation_metrics.py  # Confusion matrices and semantic segmentation evaluation
scripts/
  run_vla.sh               # Designated WSL/vla-supervised interpreter
  setup_sidewalk.sh         # Add data-reading dependencies
  segment_sidewalk.sh       # Dataset-class segmentation + evaluation
  segment_custom.sh         # Custom-class segmentation
  segment_nuscenes.sh       # nuScenes lidarseg classes + projected-point evaluation
  segment_nuscenes_custom.sh # nuScenes custom-class segmentation
  segment_nuscenes_6cams.sh # Six-view segmentation, fused point clouds, 3D results, and evaluation
  segment_nuimages.sh       # nuImages class segmentation, ground-truth comparison, and evaluation
  segment_nuimages_custom.sh # nuImages custom TXT class segmentation
  link_nuimages.sh          # Create/check the WSL dataset symlink
  setup_nuimages.sh         # Add RLE dependencies to vla-supervised (already installed here)
  test_nuimages.sh          # nuImages data-reading and evaluation pipeline tests
  color_nuscenes_pointcloud.sh # Assign existing pixel classes to point clouds, generate 3D results and evaluation
  test_sidewalk.sh          # All data-processing and evaluation regression tests
  test_nuscenes.sh          # nuScenes projection and pipeline tests
config/
  sidewalk_classes.txt     # Original dataset class names -> English SAM3 prompts
  nuscenes_classes.txt     # Official nuScenes 16 classes -> SAM3 prompts
  nuscenes_lidarseg32_classes.txt # Prompts for nonzero classes in the original 32-class labels
  nuimages_classes.txt     # Original nuImages class names -> SAM3 prompts
  classes.txt              # Example user-defined classes
```

The project uses these existing WSL symlinks:

- `models/sam3` → `/mnt/d/model_zoo/sam3`, using the official `sam3.pt`.
- `datasets/sidewalk-semantic` → `/mnt/d/datasets/sidewalk-semantic`, reading `data/train-*.parquet` and `id2label.json` directly.
- `datasets/nuscenes` → `/mnt/d/datasets/nuscenes`, using `v1.0-mini/`, `samples/`, and `lidarseg/v1.0-mini/`.
- `datasets/nuimages` → `/mnt/d/datasets/nuimages`, using 2D annotated images from `v1.0-mini/` and `samples/`.

See [Environment setup](#environment-setup) above for activating the current environment, adding dependencies, and creating a new environment. The sections below describe class configuration, outputs, and evaluation for each dataset.

## 1. Sidewalk Semantic classes and evaluation

First check the output with a few images:

```bash
bash scripts/segment_sidewalk.sh \
  --max-images 5 \
  --output-dir outputs/sidewalk_demo
```

Evaluate all 1000 training images:

```bash
bash scripts/segment_sidewalk.sh \
  --output-dir outputs/sidewalk_full
```

By default, the original 34 nonzero classes are read, preserving dataset IDs 1–34; ID 0 is `unlabeled`. `config/sidewalk_classes.txt` maps dataset label names such as `flat-sidewalk` to English prompts such as `sidewalk` that better suit SAM3. You can modify prompts to the right of `=` and add aliases with `|`; the left side must retain the complete original class name. You can also specify a complete replacement mapping with `--classes-file config/your_sidewalk_prompts.txt`.

The original classes include `void-ground`, `void-dynamic`, `void-static`, and `void-unclear`. These labels lack precise natural-language visual concepts, so the configuration uses approximate prompts, while evaluation still scores the original IDs. This is zero-shot semantic segmentation evaluation using SAM3 text prompts, without a trained semantic classification head. Prompts, confidence thresholds, and mask-conflict handling all affect the metrics.

Only the `train` split is available locally. The script also accepts `--split validation/test`, but you must provide the corresponding Parquet files in the same format. When running on a subset, metrics cover only the images actually processed and do not represent the entire dataset.

## 2. Custom TXT classes

Edit `config/classes.txt`, with one English class per line:

```text
road
sidewalk
person
car
```

You can specify a display class name separately from one or more English prompts:

```text
plants = vegetation | tree | bush
crosswalk = pedestrian crosswalk
```

Blank lines, `#` comments, and a UTF-8 BOM are supported. Custom classes receive IDs 1, 2, 3, ... in configuration order; ID 0 denotes uncovered pixels. Up to 254 classes are supported. Duplicate class names, empty prompts, and the reserved names `unlabeled`/`unassigned` produce errors.

```bash
bash scripts/segment_custom.sh \
  --classes-file config/classes.txt \
  --max-images 5 \
  --output-dir outputs/custom_demo
```

Custom-class mode still segments Sidewalk images, but does not read ground-truth labels or compute semantic metrics because custom classes may not correspond to the original labels.

## Outputs

Use a new or empty `--output-dir` for each run to avoid mixing results from different parameters. Existing nonempty output directories are not overwritten.

```text
outputs/<run>/
  predictions/train_000000.png     # Original-size, single-channel uint8 class-ID map
  color_masks/train_000000.png     # Colored semantic segmentation map
  overlays/train_000000.jpg        # Original-size image overlay
  visualizations/train_000000.jpg  # Original/prediction overlay/predicted class map/GT (dataset mode only) comparison and legend
  instances/train_000000.json      # Prompts, classes, instance scores, boxes, and mask pixel counts before merging
  classes.json                    # IDs, class names, RGB colors, and prompts
  run_config.json                 # Parameters, environment, data selection, run status, and timing
  samples.jsonl                   # Per-image index, dimensions, instance count, coverage, and processing time
  metrics.json                    # Dataset-class mode only: overall and per-class metrics
  metrics.csv                     # Dataset-class mode only: overall and per-class metrics table
  confusion_matrix.csv            # Dataset-class mode only: GT rows, prediction columns
```

`instances` stores instance metadata; final pixel-level segmentation is stored in `predictions`. Comparison visualizations are downscaled to a maximum panel width of 640, while prediction masks and overlays retain their original dimensions. ID 0 is black in color maps and preserves the original image in overlays; GT 255 is marked as ignored in the legend.

SAM3 encodes each image once and generates instance masks per class/prompt. Instances are first filtered by `--score-threshold`, then pixel probabilities are binarized by `--mask-threshold`. If multiple instances cover the same pixel, the class of the instance with the highest confidence is used. Ties retain the earlier class/prompt/instance. Uncovered pixels are set to 0. Multiple instances of the same class merge naturally into that class.

## Evaluation conventions

- Ignore **GT IDs 0 and 255**: these pixels are entirely excluded from the confusion matrix.
- On other valid GT pixels, predicted ID 0 still enters the confusion matrix and counts as a false negative for the true class. Pixels are not removed from evaluation because segmentation missed them.
- Accumulate a pixel confusion matrix over the entire selected data subset before computing metrics, rather than directly averaging per-image metrics.
- Output `pixel_accuracy` (pixel accuracy), `mean_iou` (mIoU), `mean_accuracy` (mean class recall), `frequency_weighted_iou` (frequency-weighted IoU), `macro_dice`, and per-class IoU, Dice, precision, recall, TP, FP, FN, and pixel counts.
- Class 0 is excluded from macro metrics. mIoU/Dice exclude classes absent from both GT and predictions; classes with false positives but no GT still contribute a score of 0. Mean class accuracy includes only classes present in GT.
- Metric values are in **0–1**. Metrics without a defined denominator are written as JSON `null` / empty CSV cells, not NaN. The JSON also includes formulas and the ignore policy.

After an exception or Ctrl+C, if sample processing has begun, the currently accumulated metrics are saved and `run_config.json` is marked as failed/interrupted. These are partial results; check `status` and `processed_images`.

## Common parameters and checks

```bash
# Inspect only Parquet fields, sample dimensions, and label IDs, without loading SAM3
bash scripts/segment_sidewalk.sh --inspect-data

# Process 10 images from global index 100, generating fewer visualizations
bash scripts/segment_sidewalk.sh \
  --start-index 100 --max-images 10 --vis-every 5 \
  --score-threshold 0.5 --mask-threshold 0.5 \
  --output-dir outputs/sidewalk_subset

# Show all parameters
bash scripts/segment_sidewalk.sh --help

# Regression tests: class configuration, mask merging, Parquet reading, visualization, and evaluation metrics
bash scripts/test_sidewalk.sh
```

The default `--device cuda --precision bf16` suits this machine's RTX 5070 Ti. `--precision fp32/fp16` is supported; CPU mode requires both `--device cpu --precision fp32`. `--no-vis` saves only class-ID maps, instance information, and evaluation results. `--max-images 0` processes every image from the start index to the end.

## 3. nuScenes mini: camera segmentation and lidarseg projected-point evaluation

Run only in WSL's `vla-supervised` environment, reusing local SAM3. JSON reading and point-cloud projection use NumPy/Pillow, without installing `nuscenes-devkit` or changing the existing environment.

The default uses `CAM_FRONT` from `mini_val`, processes **at most 10 images**, and uses the official 32→16 class mapping. You can begin by limiting it to 1–5 images:

```bash
bash scripts/segment_nuscenes.sh \
  --max-images 5 \
  --output-dir outputs/nuscenes_demo
```

Select one scene from the full mini dataset and multiple cameras:

```bash
bash scripts/segment_nuscenes.sh \
  --split all --scenes scene-0061 \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
  --max-images 6 \
  --output-dir outputs/nuscenes_scene0061
```

`--split` supports `mini_train`, `mini_val`, and `all`. Scene filtering is intersected with the split. The six available cameras are `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, and `CAM_BACK_RIGHT`. Only keyframes are used, associated with `LIDAR_TOP` and lidarseg labels from the same sample; sweeps are not used. Enumeration follows scene name, sample timestamp, and the specified camera order.

The local mini dataset contains 404 samples, or 2,424 images across all six cameras; mini_train has 323 samples and mini_val has 81. **`--max-images` and `--start-index` count camera images in the selected list**. For example, with six cameras, `--max-images 6` processes all six views of one sample. Only `--max-images 0` processes every selected view from the start index to the end. Each run requires a new or empty output directory.

### Class modes

- Default `--taxonomy lidarseg16`: 16 valid classes, IDs 1–16; ID 0 means ignore/uncovered. Uses `config/nuscenes_classes.txt`; retain the class names on the left, and modify prompts or add `|` aliases on the right. The classes are barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain, manmade, and vegetation.
- `--taxonomy lidarseg32`: preserves original label IDs 0–31; 0=noise is ignored, and all IDs 1–31 are prompted and evaluated. Uses `config/nuscenes_lidarseg32_classes.txt`. Text prompts for pedestrian subcategories and `vehicle.ego`, among others, have semantic ambiguities. This mode is an experiment with the original classes; the official benchmark generally uses the merged 16 classes.
- `--classes-file` can replace the complete prompt configuration for that taxonomy; class names and IDs stay the same.

```bash
# Original lidarseg 32 classes
bash scripts/segment_nuscenes.sh \
  --taxonomy lidarseg32 --max-images 5 \
  --output-dir outputs/nuscenes_raw32

# Custom TXT classes: segmentation and visualization only, without lidarseg metrics
bash scripts/segment_nuscenes_custom.sh \
  --classes-file config/classes.txt --max-images 5 \
  --output-dir outputs/nuscenes_custom
```

Class mappings are checked against the official source and written to `lidarseg_mapping.json` in each evaluation output. The local `category.json` is a detection-box category table without lidarseg indices; the script does not infer labels from it. Official references: [original class/index order](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py), [32→16 mapping](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/utils.py), and [mini split scene lists](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py).

### Projection and evaluation conventions

Each lidarseg label corresponds to one LiDAR point, rather than a pixel annotation for the entire camera image. The script projects the point cloud into the camera through this chain:

```text
LiDAR sensor -> ego at LiDAR timestamp -> global
             -> ego at camera timestamp -> camera sensor -> camera intrinsic
```

The ego pose at each sensor's timestamp is used, with quaternions interpreted as `w,x,y,z`. Points are retained when their camera depth exceeds `--min-depth` (1 m by default) and they lie inside the image with a 1-pixel margin. The full-resolution prediction map is sampled at `floor(u), floor(v)`. If multiple projected points occupy the same integer pixel, only the nearest point is retained; equal-depth ties retain the earlier original point index. Label and point counts must match, and raw labels must be in 0–31, or an error is raised.

- In 16-class mode, the official mapping ignores GT coarse ID 0, including original noise, ego, and other officially excluded classes; 32-class mode ignores only GT raw ID 0.
- Predicted ID 0 on valid GT points still counts as an error; uncovered points cannot be discarded.
- A point-observation confusion matrix is accumulated across all processed views, producing point accuracy, mIoU, mean class recall, frequency-weighted IoU, macro Dice, and per-class metrics. Macro-metric rules match those for Sidewalk.
- **This evaluates sparse projected points in camera images**. It does not measure dense image semantic segmentation or provide an official full-surround point-cloud benchmark score. Occlusion across pixels and sensor timing differences for moving objects are not fully eliminated. A LiDAR point appearing in multiple cameras is counted separately for each camera observation.
- `metrics.json` contains `point_accuracy`, `valid_point_observations`, and a detailed `evaluation_scope`. For compatibility with the shared metric implementation, it also retains the aliases `pixel_accuracy/valid_pixels` and per-class `gt_pixels/predicted_pixels`, but every unit in this mode is a **projected-point observation**. CSV uses `point_accuracy`, `gt_points`, and `predicted_points`, and specifies the unit on each row.
- Custom mode has no corresponding label mapping and does not generate evaluation metrics or projected GT. The local mini data directory still needs complete lidarseg metadata for keyframe association checks.

### nuScenes outputs

As with Sidewalk, the output includes `predictions/`, `instances/`, `overlays/`, `color_masks/`, `visualizations/`, `classes.json`, `run_config.json`, and `samples.jsonl`. Filename prefixes include the scene, selected-view index, camera, and sample token. Additional outputs:

```text
projected_points/<stem>.npz      # point_indices, uv, pixels, depth, raw_labels, target, prediction
sparse_gt/<stem>.png            # Original-size uint8 sparse GT; unannotated pixels are 255
point_overlays/<stem>_gt.jpg     # Colored lidarseg GT points overlaid on the original image
point_overlays/<stem>_prediction.jpg # Predicted classes overlaid at the same projected points
lidarseg_mapping.json           # Mapping from each raw ID to the current taxonomy
metrics.json / metrics.csv / confusion_matrix.csv
```

Comparison images show the original image, SAM3 overlay, dense predicted class map, and **sparse LiDAR GT point overlay**. Sparse annotations are not filled in to form dense GT. `--no-vis` disables visualization but still saves class maps, projected points, sparse GT, and evaluation. After an exception/interruption, check `status/processed_images` in `run_config.json`; partially accumulated metrics do not represent a complete run.

```bash
# Inspect selected images and actual projected-point counts without loading the model
bash scripts/segment_nuscenes.sh --inspect-data

# Show all parameters
bash scripts/segment_nuscenes.sh --help

# Tests for projection, classes, sparse visualization, and metrics
bash scripts/test_nuscenes.sh
```

## 4. Color original LiDAR point clouds with pixel classes

Pixel predictions are assigned back to the **original point indices** using the same calibrated projection. Each LiDAR point stores its predicted class, GT, coverage status, and color. Coordinates remain in the original `LIDAR_TOP` sensor frame, in meters. Points not projected into a processed camera are gray; covered points without a class assigned by SAM3 are light gray. Other points use the pixel-class palette, matching the class colors in the 2D images.

### Export existing segmentation results without rerunning inference

```bash
bash scripts/color_nuscenes_pointcloud.sh \
  --input-dir outputs/nuscenes_demo \
  --output-dir outputs/nuscenes_demo_cloud
```

The input is a completed nuScenes segmentation output directory; the script reads `samples.jsonl`, `predictions/`, and the configuration. Images recorded as completed in a failed/interrupted run can also be exported, with the source status preserved in the output. Do not export while the source segmentation task is still running. Custom-class mode can also export predicted point clouds and coverage, but does not compute class metrics against lidarseg.

`--max-frames 1` exports just one LiDAR keyframe; the default 0 exports all completed point-cloud frames in the input. `--dataset-dir` can override the data root in the source configuration. Use a new or empty output directory to avoid mixing results from different inference runs.

### Run six-camera segmentation, coloring, and evaluation together

The dedicated six-view entry point processes one keyframe (6 camera images) by default and automatically exports the point cloud:

```bash
bash scripts/segment_nuscenes_6cams.sh --output-dir outputs/nuscenes_six_demo
```

To process 10 frames, specify `--max-images 60`. Multiples of 6 are recommended for `--max-images` and `--start-index` to avoid having only some views for the first or last frame. You can also use the general entry point:

```bash
bash scripts/segment_nuscenes.sh \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \
  --max-images 6 \
  --export-pointcloud \
  --output-dir outputs/nuscenes_one_scan
```

Here, 6 images correspond to one sample's six cameras and are fused into one colored point-cloud frame. The default is `--split mini_val --taxonomy lidarseg16`; the original `lidarseg32` or custom-class entry point can also be used. Add `--export-pointcloud` to the existing segmentation command to export automatically to `<output-dir>/pointcloud_results/`. Existing single-camera results cover only that camera's region; exporting a point cloud does not automatically run the other cameras. If `--start-index`/`--max-images` truncates the camera list, only the views actually completed for that frame are fused, and the camera list is recorded in the output.

When the same original point appears in multiple cameras, fusion uses a **majority vote among nonzero classes**. Class 0 means segmentation did not assign a class and is treated as an abstention, but the point still counts as geometrically covered. Ties favor the earlier camera in the input list. Each point is evaluated once after fusion to avoid counting camera-overlap regions repeatedly. Fusion does not read GT or use ground truth to select predictions.

### View and use point-cloud files

```text
<pointcloud-output>/
  index.html                  # Entry point for each frame's offline 3D view
  run_config.json             # Source, status, and number of processed point-cloud frames
  frames.jsonl                # Per-frame camera lists and coverage statistics
  evaluation/
    metrics.json / metrics.csv # Unique points: covered_points and full_scan metric sets
    coverage.csv              # Per-class GT coverage
    confusion_matrix_covered.csv
    confusion_matrix_full_scan.csv
  pointclouds/<scene_frame_sample>/
    viewer.html               # Interactive rotate/zoom/pan; switch among prediction/GT/errors/coverage
    preview.png               # Static top-down and oblique comparison views
    prediction.ply            # All points, colored by predicted class; uncovered points are gray
    prediction_covered.ply    # Camera-covered points only, including prediction0 points
    ground_truth.ply          # Full-frame GT class colors, dataset-class mode only
    errors.ply                # Covered points with nonzero GT: correct green, incorrect red; others dark gray
    points.npz                # Complete original point sequence, classes, coverage, fusion sources, and other arrays
    summary.json              # Current-frame statistics, coverage, and evaluation
    evaluation/               # Two metric sets for the current frame, dataset-class mode only
```

`viewer.html` and `index.html` are self-contained and can be opened directly in a browser without a web service or network access. The browser must support WebGL. Drag with the left button to rotate, right button/Shift to pan, and the scroll wheel to zoom. You can show only covered points and adjust point size. By default, HTML draws at most 60,000 points; larger clouds are sampled deterministically and uniformly from the original point sequence. The interface shows the actual number drawn. Adjust this with `--html-max-points`. **PLY, NPZ, PNG, and evaluation all retain the full set of points**, so HTML display sampling does not change metrics; no points are cropped from the full frame.

PLY files can be opened with CloudCompare, MeshLab, or another PLY reader. They retain the `class_id`, `covered`, and `source_camera` properties. source_camera is an index into camera_names in `points.npz`, and is -1 when the prediction is 0. NPZ contains `xyz/intensity/ring/raw_labels/point_indices/prediction/covered/observation_count/winning_votes/source_camera/camera_names`, plus `target` in dataset-class mode. Arrays follow the original LiDAR file's point order.

### Point-cloud evaluation conventions

- `covered_points`: evaluates only unique points with geometric projection coverage from at least one processed camera, ignoring GT0. A covered point with prediction0 is still an error; selecting only successfully colored points cannot inflate the metrics.
- `full_scan`: includes every nonzero GT point in the entire frame, with prediction0 on uncovered points counted as an error. It reflects the ability to assign classes throughout the full frame; this score is typically much lower for a single camera.
- `coverage`: reports total points, covered points, points with actual nonzero predictions, valid GT coverage, and per-class coverage. Metrics are available per frame and across all exported frames; the GT merging policy still follows the source configuration's 16/32-class mode.
- Both scopes output point accuracy, mIoU, mean accuracy, frequency-weighted IoU, macro Dice, and per-class precision/recall/IoU/TP/FP/FN. A prediction of 0 creates an FN for the corresponding GT class; GT0 is entirely excluded from macro metrics and the confusion matrix. The same point across multiple cameras is counted once, and point indices are independent for each frame.
- Calibration, occlusion across pixels, and timing differences for dynamic objects still introduce errors. Point-cloud results come from camera predictions; they are not outputs of a trained 3D semantic segmentation network or official nuScenes benchmark submission scores.

`--no-html` / `--no-preview` disable interactive files and static images, respectively. In a segmentation command, `--export-pointcloud --no-vis` exports PLY/NPZ and evaluation while disabling PNG/HTML. Point-cloud metrics are in the point-cloud output's `evaluation/` directory; `metrics.json` at the original segmentation output root still counts projected-point observations per camera. The two evaluation scopes differ.

## 5. nuImages: image segmentation, ground-truth comparison, evaluation, and visualization

The local data at `D:\datasets\nuimages` is **v1.0-mini, with 50 annotated keyframes**. The WSL symlink `datasets/nuimages -> /mnt/d/datasets/nuimages` has been created. You can rerun the following command to check/create the link; an existing different link or directory will not be overwritten:

```bash
bash scripts/link_nuimages.sh
```

The current WSL `vla-supervised` environment already includes SAM3, PyTorch, Pillow, NumPy, and `pycocotools` for RLE decoding; no additional devkit is needed. If another environment with the same name lacks the decoding dependencies, run `bash scripts/setup_nuimages.sh`. The additional dependencies are recorded in `requirements-nuimages.txt`.

### Dataset-class mode

```bash
bash scripts/segment_nuimages.sh \
  --max-images 5 \
  --output-dir outputs/nuimages_demo
```

The default is `--version v1.0-mini`, processing at most 10 images and selecting **annotated key cameras** from all camera channels. The local mini keyframes are distributed as follows: front 8, front-left 5, front-right 7, back 11, back-left 10, and back-right 9. Each nuImages sample provides one annotated key camera and unannotated sweeps before and after it from that camera, unlike the six synchronized cameras in a nuScenes sample. The script does not treat sweeps as samples with GT or invent shared annotations for six cameras.

Filter cameras, for example to compare only front views:

```bash
bash scripts/segment_nuimages.sh \
  --cameras CAM_FRONT --max-images 5 \
  --output-dir outputs/nuimages_front
```

`--cameras` accepts multiple names from the six camera channels; the default applies no filter. Images are sorted by image timestamp and sample token; `--start-index` and `--max-images` refer to the filtered keyframe list. The default processes only 10 images; only `--max-images 0` processes every selected image from the start index to the end. The output directory must be new or empty.

`--version v1.0-train/v1.0-val` supports data in the same format, but you must supply the corresponding metadata and images; only mini is currently available locally. `v1.0-test` has no available ground truth, so the script allows inference only in custom-class mode.

### Classes and ground-truth composition

Uses `config/nuimages_classes.txt`: retain original class names on the left, and modify English prompts or add aliases with `|` on the right. You can also specify a complete replacement prompt file with `--classes-file`. Official IDs are preserved:

- ID 0: unannotated background; object classes are ordered alphabetically as 1–23.
- ID 24: `flat.driveable_surface`; ID 31: `vehicle.ego`.
- IDs 25–30: unused numbers retained as placeholders, without prompting or evaluation.
- By default, ego is neither prompted nor evaluated, giving 24 active classes for inference. Adding `--include-ego` prompts and evaluates ID 31, for a total of 25 classes.

These IDs **cannot directly reuse the nuScenes lidarseg class mapping**. Each run's `classes.json` records names, IDs, active status, prompts, and colors.

Annotation masks are base64-wrapped COCO compressed RLE, decoded from base64 before passing them to pycocotools; their dimensions must match the image. Surface masks are drawn first in JSON order, then object masks sorted by object annotation token, so objects overwrite surfaces, following the official `get_segmentation()` rules. Null masks are skipped and counted; masks are not fabricated from detection boxes. Uncovered pixels are GT0.

Official implementation references: [mask_decode / name_to_index_mapping](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuimages/utils/utils.py) and [get_segmentation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuimages/nuimages.py).

### Evaluation scope and outputs

By default, GT0, GT31, the dataset's unused IDs 25–30, and 255 are ignored. Prediction0 within valid GT regions still counts as an error. A pixel confusion matrix is accumulated over all processed images, producing mIoU, pixel accuracy, mean accuracy, frequency-weighted IoU, macro Dice, and per-class IoU/Dice/precision/recall/TP/FP/FN. Macro-averaging follows the same rules as other image modes, and metric values are in 0–1.

**nuImages annotations do not cover the complete set of scene semantic classes**: unannotated background regions are ignored, and GT classes such as sky, buildings, and vegetation needed for this experiment are not provided. Metrics reflect only valid annotated object/surface pixels and cannot be interpreted as scene segmentation performance across all image pixels. The `evaluation_scope` in `metrics.json` records total image pixels, annotated pixels, the valid GT fraction, mask counts, the ignore policy, and run status. This implementation computes semantic pixel metrics, not instance segmentation AP.

```text
outputs/<nuimages_run>/
  predictions/<stem>.png         # Original-size uint8 class-ID predictions
  ground_truth/<stem>.png        # Original-size uint8 semantic GT composed by official rules
  instances/<stem>.json          # SAM3 prompt/instance score/class/box and other metadata
  visualizations/<stem>.jpg     # 3 columns x 2 rows: original, prediction overlay, prediction map, GT overlay, GT map, errors
  overlays/<stem>.jpg           # Prediction overlay
  color_masks/<stem>.png        # Colored prediction map
  gt_overlays/<stem>.jpg        # Ground-truth overlay
  gt_color_masks/<stem>.png     # Colored ground-truth map
  errors/<stem>.png             # Correct green, incorrect red, ignored/unannotated gray
  classes.json / run_config.json / samples.jsonl
  metrics.json / metrics.csv / confusion_matrix.csv
```

`--no-vis` disables color maps, overlays, error maps, and comparison images, but still saves predictions, ground truth, and evaluation. `--vis-every N` reduces visualization frequency. On interruption/failure, accumulated metrics are saved and the status is marked; check `status/processed_images` in `run_config.json`.

### Custom TXT class mode

```bash
bash scripts/segment_nuimages_custom.sh \
  --classes-file config/classes.txt --max-images 5 \
  --output-dir outputs/nuimages_custom
```

Classes receive IDs 1..N in TXT order, with 0 meaning unassigned, as in the other custom entry points. This mode reads only image metadata and does not require annotation tables. It outputs predictions, color maps, overlays, and three-panel comparisons, without GT/error maps or semantic evaluation, because custom classes may not correspond to the annotations.

```bash
# Inspect image selection, annotation decoding, class IDs, and valid-pixel counts, without loading SAM3
bash scripts/segment_nuimages.sh --inspect-data

# Show parameters or run nuImages regression tests
bash scripts/segment_nuimages.sh --help
bash scripts/test_nuimages.sh
```
