# vla-supervised-ad

**English** | [简体中文](guide.zh-CN.md)

Use local SAM3 for text-prompted segmentation of camera images from Sidewalk Semantic, nuScenes mini, and nuImages, and export class maps, visualizations, and evaluation metrics. nuScenes supports lidarseg projection evaluation, fusion of original LiDAR points, and 2D BEV export; nuImages is evaluated against image-mask ground truth.

Suggested reading order: [Environment setup](#environment-setup) → [Command reference](#command-reference) → [Project layout and data paths](#project-layout-and-data-paths). The following sections describe each dataset, point-cloud outputs, and evaluation conventions in detail.

## Instance minimum rectangles

The scene viewer adds a default-hidden Instance boxes layer in3D/BEV Map, following frame/all-frame/class selection independently of point and surface layers. Each camera-local instance uses the minimum-area rotated rectangle of retained support XY, preserving empty/point/segment cases. Corners, dimensions, yaw, area and original point indices are saved; boxes are neither detector GT nor track IDs. New exports include them automatically, and saved instance outputs can be re-exported without inference. See [README instance boxes](../README.md#minimum-area-bounding-boxes-per-instance).

## Instance SOR/DBSCAN cleaning

The generator supports none/sor/dbscan/sor_dbscan; CLI uses `--point-filter` plus SOR neighborhood/std, DBSCAN radius/core count and cluster-retention parameters. Actual camera-local instance memberships are filtered, original arrays remain saved, and retained support drives point polygons and colored semantic points while IPM stays unchanged. Older exports lacking instance membership require segmentation regeneration. See [README instance cleaning](../README.md#instance-support-cleaning-sor--dbscan) for parameters, cluster policies and before/after evidence. Historical no-outlier-removal descriptions below defer to this optional configuration; default is off.

## Local generator UI

Run `bash tasks/nuscenes_sam3/scripts/run_generator.sh` (or run_generator.bat on Windows), then open http://127.0.0.1:8765. Edit categories/prompts, choose plane classes and mini scenes, preview/start generation, inspect logs/progress, stop a run and open results. Each run saves an independent configuration without changing the default TXT. See [README graphical generator](../README.md#graphical-generator).

## Configurable classes and plane selection

The custom pipeline is no longer fixed to six categories. Add crosswalk or other names to the prompt TXT; IDs follow non-comment line order. Select surfaces with `--plane-classes sidewalk road crosswalk`; unselected categories remain colored points. The parameter works throughout run_scenes.sh, run_six_classes.sh and flat_ground. BEV Map and HTML use each output's saved dynamic schema and plane_class_ids. Old filenames remain compatibility entries; historical six-class/road-sidewalk-only descriptions below defer to the saved schema and current selection. See [README plane selection](../README.md#choose-plane-classes-including-crosswalk) for commands and old-ID protection.

## BEV Map grid mode

The scene HTML's **BEV Map** button samples each ground-surface method into the same metric grid, supporting current/all-frame vote fusion, method selection, class hiding and original point XY overlays. Unassigned cells stay blank. Default0.2m/cell; labels, per-class frame votes, coverage counts and conflicts are saved without reclassifying unsupported cells as free space. New exports include it automatically; upgrade saved scenes with `bash tasks/nuscenes_sam3/scripts/run_bev_map.sh --input-dir <existing-scenes> --output-dir <new-directory>`. See [README BEV Map](../README.md#bev-map-divide-ground-surfaces-into-metric-cells) for options and files.

## Interactive scene timelines

The six-class pipeline now exports one standalone3D HTML per scene, with zero-plane IPM and BEV point-polygon road/sidewalk surfaces, original colored semantic points, playback/all-frame mode and layer buttons. See the [README scene section](../README.md#scene-html-two-road-surfaces-colored-points-and-a-timeline) for usage and coordinate definitions. Reuse saved six-class results with `bash tasks/nuscenes_sam3/scripts/run_scenes.sh --input-dir <saved-six-class-output> --output-dir <new-directory>` without new segmentation. Scene HTML embeds the complete data without a CDN; the index distinguishes partial and complete scenes.

## One command per task

For normal use, pick **one command below**. Each entry point automatically selects the existing WSL `vla-supervised` environment and finishes segmentation, applicable projection/evaluation, and comparison images. You do not need to run the advanced stages separately.

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad
```

| Task | Run | Final comparison |
|---|---|---|
| EFFOcc + SAM3 | `bash tasks/effocc/scripts/run.sh` | Same six cameras: original / SAM3 / EFFOcc occupancy overlay |
| Flat-ground camera-only projection (new task) | `bash tasks/flat_ground/scripts/run.sh` | SAM3 road/sidewalk regions directly inverse-projected onto global z=0; no LiDAR |
| Six requested classes + road/sidewalk planes | `bash tasks/nuscenes_sam3/scripts/run_six_classes.sh` | sidewalk / road / pedestrian / vehicle / bike / rider, six cameras and 500x500 BEV |
| nuScenes SAM3 | `bash tasks/nuscenes_sam3/scripts/run.sh` | Six-camera SAM3 / sparse lidarseg GT, plus 500x500 semantic BEV comparison |
| Continuous road/sidewalk planes (saved frame) | `bash tasks/nuscenes_sam3/scripts/run_ground.sh` | Merge six-camera ground regions; sparse / filled / source-map comparison |
| SAM3 instance polygons (nuScenes mini experiment) | `bash tasks/nuscenes_sam3/scripts/run_instances.sh` | One six-camera frame: instance masks / assigned LiDAR points / minimum convex BEV hulls |
| nuImages | `bash tasks/nuimages/scripts/run.sh` | Original / SAM3 / dataset GT / error map |
| Sidewalk | `bash tasks/sidewalk/scripts/run.sh` | Original / SAM3 / dataset GT |

**Default: one image for Sidewalk/nuImages, or one six-camera frame for nuScenes/EFFOcc.** Every run creates `outputs/<task>/<timestamp>/`, so repeat runs do not overwrite previous results. Open the printed **`result.html`** in a browser; the final comparison pictures are collected in **`results/`**. Numeric masks, intermediate predictions, and detailed metadata remain in `raw/`; EFFOcc's SAM3 comparisons also remain in `comparison/`.

**nuScenes BEV is part of the same one-command output.** `bash tasks/nuscenes_sam3/scripts/run.sh` now generates both the original semantic/GT BEV and continuous road/sidewalk planes. Open `result.html` and use the **BEV** section at the top; all six per-frame BEV/ground PNGs are also collected in `results/BEV/<frame>/`: `bev_prediction.png`, `bev_ground_truth.png`, `bev_comparison.png`, `ground_plane.png`, `ground_sources.png`, and `ground_comparison.png`. Six-camera comparisons follow in a separate section. Missing required BEV products fail the result collection instead of silently returning only camera images.

Older completed one-command runs can be upgraded without rerunning SAM3:

```bash
bash tasks/nuscenes_sam3/scripts/run.sh \
  --refresh-results \
  --output-dir outputs/nuscenes_sam3/20261016_102509_921512
```

Use the actual existing output directory for your run. Refresh adds missing ground outputs from saved point predictions, keeps original predictions/metrics, and rebuilds the result index. A fresh run still uses the plain `run.sh` command. Ground interpolation follows the already-fused point classes in the standard semantic pipeline; the separate instance experiment resolves classes by mask score before interpolation. Both record their source and neither uses GT for filling.

Only change these options when needed:

```bash
# Three six-camera frames, with an explicit new output directory.
bash tasks/effocc/scripts/run.sh --samples 3 --output-dir outputs/my_effocc_comparison

# Begin at frame 1, which has real preceding LiDAR sweeps.
bash tasks/effocc/scripts/run.sh --start-index 1
```

`--samples` always counts complete six-camera frames for nuScenes/EFFOcc; the launcher converts to image counts internally. `--start-index` uses the same task-specific unit. Default device is CUDA; `--device cpu` selects the compatible float32 SAM3 mode. The EFFOcc stages run in separate processes to release its model memory before loading SAM3.

EFFOcc compares two independent model predictions; the other tasks compare SAM3 with the dataset's available GT. This launcher does not add a second segmentation model to Sidewalk/nuImages, invent dense image GT from lidarseg, or claim an accuracy score for EFFOcc versus SAM3. Existing dataset metrics are collected alongside the final images.

The following sections document environment setup and **advanced individual stages** for changing classes, data paths, or other settings. The original commands remain supported.

## Flat-ground IPM: project semantic regions directly onto z=0

This is an independent task in [tasks/flat_ground](../tasks/flat_ground/), separate from LiDAR semantic rasterization and triangle/hull surface filling. It assumes a horizontal plane at **nuScenes global-map z=0 meters**, not geodetic sea-level altitude. The display frame is centered at the current CAM_FRONT ego XY position, retaining only ego yaw: local +X is forward and +Y is left. PNG columns increase X, rows decrease Y. Default ROI is100mx100m with0.2m cells,500x500.

```bash
# Run SAM3 on one six-camera frame, then directly project road/sidewalk masks.
bash tasks/flat_ground/scripts/run.sh

# Reuse already saved six-class masks, without running SAM3 or reading point clouds.
bash tasks/flat_ground/scripts/run.sh \
  --input-dir outputs/nuscenes_six_class_planes \
  --plane-z 0 \
  --output-dir outputs/flat_ground_demo
```

Default output is `outputs/flat_ground/<timestamp>/result.html`, with a combined BEV/coverage/source-camera comparison and six camera panels (original / SAM3 / per-camera IPM). `--samples 1`, `--start-index`, `--split mini_val/mini_train/all`, `--scenes`, `--plane-z`, `--bev-roi-size`, `--bev-resolution`, `--max-distance` (80m camera-to-ground slant range), and `--output-dir` are available. With `--input-dir`, indices select saved mask frames and exact sample/camera tokens are checked. Without it, SAM3 uses the six-class configuration and local checkpoint. Camera metadata is read independently: LiDAR samples, sweeps and lidarseg tables are not required.

For a camera pixel ray, the plane intersection is `p_global = camera_origin + t * ray_global`, where `t = (plane_z - camera_origin_z) / ray_global_z`. Parallel, backward/behind-camera, too-near and overly distant intersections are excluded. To avoid holes from forward pixel splatting, the implemented IPM evaluates each BEV **cell center**, projects its global z=plane_z point into each original camera, and samples the nearest semantic pixel. This is inverse sampling of the same plane homography, with no point-cloud depth or learned depth. Non-ground labels do not become road; unassigned masks or no valid camera correspondence remain white.

By default only `sidewalk` and `road` are mapped. Use `--project-classes sidewalk road pedestrian vehicle bike rider` only if intentionally flattening every category: elevated people/vehicles/bikes then have distorted ground silhouettes, not real ground footprints. In overlapping camera views, an accepted class from the greatest ground-ray incidence cosine is selected (a more fronto-parallel/closer plane view); equal scores preserve camera order. `camera_count` records geometric coverage separately from class evidence. No convex hull, triangle interpolation or extra hole filling is applied.

```text
<flat-ground-output>/
  result.html / run_config.json
  <scene_frame_sample>/
    bev_ground_ipm.png            # 500x500 assumed-ground semantic map
    bev_class_ids.png             # IDs0..6, 0 unassigned/invalid
    bev_camera_coverage.png       # Geometric camera view count (1..6)
    bev_source_camera.png         # Camera providing the selected semantic evidence
    bev_comparison.png
    ipm_grid.npz                  # Labels, source_camera, camera_count, support_count, incidence, XY grid and frame transform
    ipm_metadata.json / classes.json
    cameras/<camera>_ipm.png
    cameras/<camera>_class_ids.png / <camera>_source.json
    cameras/<camera>_sampled_bev_ids.png
    cameras/<camera>_geometry.npz # Per-cell pixel_xy/uv/depth/slant_distance/incidence/valid, calibration and pose
    cameras/<camera>_comparison.jpg
```

**This projection is a flat-ground hypothesis, not reconstructed surface or GT.** Real slopes, raised sidewalks, occlusion and pose/mask errors cause distortion; the global z=0 convention does not guarantee the actual road lies there. Original images retain each camera's timestamps and pose. Unobserved/invalid cells stay blank, geometric visibility is not proof of unobstructed ground, and no projection accuracy or traversability score is claimed. The original LiDAR-backed tasks remain unchanged.

## Six-class segmentation with road and sidewalk planes

Use this separate entry point when only six classes are required; the existing nuScenes16 and EFFOcc workflows remain available:

```bash
bash tasks/nuscenes_sam3/scripts/run_six_classes.sh
```

Default: one complete six-camera mini_val frame, 100mx100m ROI, 0.2m cells (500x500). The printed `outputs/nuscenes_six_classes/<timestamp>/result.html` shows the semantic BEV first and each camera's original/overlay/class map. Change sample count with `--samples`, starting frame with `--start-index`, or use a new `--output-dir`.

| ID | Output class | Definition / default prompt family |
|---|---|---|
| 0 | unassigned | No accepted mask; white in the color map |
| 1 | sidewalk | Pedestrian sidewalk surface |
| 2 | road | Road surface, not lane-marking instances |
| 3 | pedestrian | Person on foot; the spelling is pedestrian |
| 4 | vehicle | Car, truck, bus and trailer; excludes bikes |
| 5 | bike | Bicycle or motorcycle; rider person is a separate class |
| 6 | rider | Person riding a bicycle or motorcycle |

Edit [six_classes.txt](../tasks/nuscenes_sam3/config/six_classes.txt), retaining left-side names and order; change English prompts after `=` and separate alternatives with `|`. Only these six concepts are prompted. Normal overlap uses instance confidence; accepted rider masks take priority over pedestrian/unassigned pixels only, without overriding a bike or vehicle label. Text prompts can still confuse rider/person/vehicle boundaries; the saved class maps and instance metadata expose these decisions.

The six views are projected onto the original LiDAR scan and fused by nonzero semantic majority per unique point. Only **sidewalk1 and road2** are interpolated with the existing bounded ground triangulation (`--max-edge 6`, meters). Other classes retain their observed semantic cells and veto interpolation at their support cells. The final plane preserves pedestrian/vehicle/bike/rider cells and changes only surface/unassigned/empty cells supported by road/sidewalk triangles. It does not expand objects or fill all unseen space. Blue in `plane_sources.png` marks inferred fill, not measured surface or safe-traversability evidence.

Each saved frame contains `six_class_comparison.png` (sparse BEV / ground-plane BEV / source map), `bev_six_sparse.png`, `bev_six_plane.png`, `road_sidewalk_plane.png`, `plane_sources.png`, `six_class_plane.npz`, `plane_metadata.json`, complete `points.npz`, and `cameras/<channel>_class_ids.png/_semantics.png/_overlay.jpg/_comparison.jpg/_instances.json/_source.json`. Public labels always use IDs0..6 (255 means no-point grid cell); an internal nuScenes16 helper mapping is used only for triangulation and is not an output taxonomy.

No unchanged nuScenes16 or Occ3D metric is reported: the requested vehicle/bike classes are grouped differently, and lidarseg has no independent rider label. Formal evaluation requires an explicit GT regrouping and rider annotation policy. Unrequested visual concepts remain unassigned; six classes are the available labels, not a guarantee that all six occur in every frame. Grid and triangulation options are available via `--help`; default SAM3 thresholds are 0.5 and GPU precision is bf16.

## Road and sidewalk: continuous ground regions across cameras

Ground classes are fused as semantic regions, not vehicle/pedestrian instance identities. Same-class points from all six cameras are deduplicated in native LiDAR XY, retaining road11, other-flat12, sidewalk13 and terrain14. Conflicting shared-point classes use mask confidence, breaking ties by the lower original instance ID. Different ground classes stay distinct.

Convert existing instance results into planes without rerunning SAM3:

```bash
bash tasks/nuscenes_sam3/scripts/run_ground.sh \
  --input-dir outputs/nuscenes_instance_hulls_experiment \
  --samples 1 --max-edge 6 \
  --output-dir outputs/ground_plane_demo
```

New `bash tasks/nuscenes_sam3/scripts/run_instances.sh` runs also create these outputs under each frame's `ground/` and show them first in `result.html`. Adjust `--ground-max-edge 6` or disable with `--no-ground-plane`; the independent converter names the same distance parameter `--max-edge`.

Ground evidence is merged into the 500x500 grid at 0.2m resolution. A per-class Delaunay triangulation over observed cell centers fills only triangles whose **every edge is at most 6m**. This locally supported surface does not stretch one class-wide convex hull across all unknown space. Existing other ground-class cells are protected; cells containing non-ground semantic points veto ground fill. Conflicting inferred regions choose the nearest same-class support cell, breaking exact ties by lower class ID.

Outputs include `ground_observed.png` (fused sparse ground), `ground_plane.png` (continuous semantic regions), `ground_sources.png` (green=observed cells, blue=inferred fill, red=non-ground evidence veto), `ground_comparison.png`, `ground_grid.npz` (observed/filled/inferred/blocked and counts), `semantic_points.npz` (deduplicated point labels), and `ground_regions.json` (per-class contours, holes, accepted triangles and metadata). The grid remains 100mx100m, 500x500. Connected road/sidewalk regions become planar patches, while distant unsupported regions are not forced into a single polygon.

**Filled surfaces are not ground-truth observations.** Single-frame support is sparse; even bounded triangles can cross unseen obstacles. Ground height/slope and traffic rules are not estimated. The experiment preserves a source mask for interpolation, makes no safe-traversability claim, and does not fabricate a dense surface accuracy metric from sparse labels.

## SAM3 instance-point convex-hull experiment

SAM3 produces a mask for each text-prompt detection. The existing semantic pipeline merges those masks; this experiment retains them and assigns the same keyframe's projected LiDAR points to individual detections. It then computes the **minimum convex polygon containing every assigned point in the 100mx100m XY ROI**. This is a convex hull, not a minimum-vertex arbitrary polygon or a fitted 3D bounding box.

```bash
bash tasks/nuscenes_sam3/scripts/run_instances.sh
```

The entry point defaults to **one complete six-camera mini_val frame**, the nuScenes16 prompts, a 100mx100m ROI and 0.2m display resolution (500x500). Results are in `outputs/nuscenes_instance_hulls/<timestamp>/result.html`. Optional `--start-index`, `--samples`, `--classes-file`, `--score-threshold`, `--mask-threshold`, `--bev-roi-size`, `--bev-resolution` and `--output-dir` are available. The TXT must retain the nuScenes16 class names. To reproduce a single frame in a named new output directory:

```bash
bash tasks/nuscenes_sam3/scripts/run_instances.sh \
  --samples 1 --bev-resolution 0.2 --bev-roi-size 100 \
  --output-dir outputs/instance_hulls_demo
```

The original per-detection masks are streamed from shared SAM3 inference before semantic merging. Within one camera, a pixel/point covered by several masks goes to the highest-confidence detection; exact ties favor the earlier detection. Pre-merge masks remain available in packed-bit form. Point assignment uses calibrated LiDAR-to-camera projection, each sensor's ego pose/timestamp, and nearest-depth selection for same-pixel collisions. **Instances start as camera-local detections, not tracks.** The original IDs/masks and shared points are preserved. The experiment now also saves a separate constrained shared-point association result: only same-class foreground instances in different cameras are candidates, and this remains an identity hypothesis rather than verified tracking.

Foreground classes1..10 (vehicles, pedestrians and other movable foreground categories) form the main hull image. Classes11..16 (road, sidewalk, terrain, manmade and vegetation regions) are included in a separate all-regions image. These SAM3 masks are semantic regions and do not establish physical object identities. Instance colors, rather than class colors, match between camera views and BEV.

```text
<instance-output>/
  result.html / run_config.json
  <scene_index_sample>/
    bev_instance_points.png       # Assigned points colored by camera-local instance ID
    bev_instance_hulls.png        # Foreground convex hulls, observed points and ID labels
    bev_all_regions.png           # Foreground plus semantic stuff-region hulls
    bev_instance_comparison.png   # Raw points / assigned points / foreground hulls / all regions
    instance_hulls.json           # Camera/instance/class/score, original point indices, hull XY and area
    points.npz                    # Original xyz/raw labels, 6xN camera_instance_ids and camera_names
    summary.json
    per_camera/<camera>_hulls.png
    cameras/<camera>_comparison.jpg      # Original / instance masks / assigned LiDAR points
    cameras/<camera>_instances_overlay.jpg
    cameras/<camera>_semantic_ids.png
    cameras/<camera>_instances.json
    cameras/<camera>_instances.npz       # Packed original masks, image instance map, projected points/pixels/IDs
```

Decode a raw mask using `np.unpackbits(packed_masks[i], count=H*W, bitorder='little').reshape(H,W)`, where `H,W` come from `mask_shape_hw`. Hull vertices are in native LiDAR XY meters and preserve the input coordinates. Three non-collinear support points can form a polygon; a single point or collinear group remains a point/segment with zero area. Detections without assigned ROI points remain empty records and are not drawn as filled objects.

**Limitations:** a convex hull fills unobserved interiors and can bridge gaps; it is not new LiDAR evidence, occupancy ground truth or a guaranteed object footprint. Projection occlusion across different pixels, mask errors and background points may stretch an instance hull considerably. No outlier removal or geometric clustering is silently applied. The shared-point merge described below is recorded separately and can be disabled; it does not repair stretched hulls or prove physical identity. The local lidarseg labels provide semantics but no matching SAM3 instance identity, so no instance/hull accuracy metric is fabricated.

### Cross-camera merging using shared LiDAR point indices

`run_instances.sh` now also produces `merged/` before/after comparisons by default. Original masks, point memberships and hulls are retained. Use `--no-merge-overlap` to disable association. Defaults are **at least 3 shared original points** and **intersection / smaller point-set size >= 0.30**; adjust with `--min-shared-points` and `--min-overlap`.

Only nonempty, same-class foreground IDs1..10 in different cameras are compared. Matching uses full-scan original point indices, not BEV polygon overlap or GT. Each pair record contains set sizes, shared points, union size, point IoU, smaller-set overlap, threshold eligibility and decision. Eligible pairs are processed deterministically by descending overlap, IoU, shared count and minimum confidence. A group may contain **at most one detection per camera**, and **every pair of members must pass** the thresholds (complete-link), so a weak A–B–C chain is not automatically fused. Class11..16 stuff and unmatched/empty instances remain separate.

A fused group's ID/color is the smallest member ID; its point set is the deduplicated union, its score is the maximum member score (not a combined probability), and its ROI convex hull is recomputed. No point is added or silently removed. Different unassociated groups can still share points. This is a single-scan experiment and assigns no cross-frame track identity.

Saved instance results can be merged without rerunning SAM3:

```bash
bash tasks/nuscenes_sam3/scripts/merge_instances.sh \
  --input-dir outputs/nuscenes_instance_hulls_experiment \
  --samples 1 --min-shared-points 3 --min-overlap 0.3 \
  --output-dir outputs/nuscenes_instance_merge_demo
```

The independent entry point writes `result.html` and one subdirectory per saved frame. Automatic merging places the same files in `<frame>/merged/`: `merge_comparison.png` (before/after foreground), `matched_groups.png` (per-match XY zoom), `bev_before_merge.png`, `bev_after_merge.png`, `bev_all_groups.png`, `merged_hulls.json`, `match_pairs.json/csv`, `instance_mapping.json`, `merged_points.npz`, and `summary.json`. The NPZ preserves `camera_instance_ids` and adds `camera_merged_ids`; group-level point indices in JSON are deduplicated. Matching uses evidence thresholds, not an accuracy guarantee; a shared background point or erroneous mask can still create a false association. The earlier convex-hull caveats remain.

## Task navigation

| Task directory | Scope | Example command (from project root) |
|---|---|---|
| [Sidewalk](../tasks/sidewalk/) | Semantic segmentation and pixel evaluation | `bash tasks/sidewalk/scripts/segment_sidewalk.sh --help` |
| [nuImages](../tasks/nuimages/) | Annotated image segmentation and pixel evaluation | `bash tasks/nuimages/scripts/segment_nuimages.sh --help` |
| [nuScenes SAM3](../tasks/nuscenes_sam3/) | Camera segmentation, projected points, fusion, and 2D BEV | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --help` |
| [EFFOcc](../tasks/effocc/) | Fusion-R18 occupancy, camera projection, and SAM3 comparison | `bash tasks/effocc/scripts/run_effocc_mini.sh --help` |
| [Flat ground](../tasks/flat_ground/) | Camera-only flat-plane semantic IPM, without LiDAR | `bash tasks/flat_ground/scripts/run.sh --help` |
| [Shared utilities](../shared/) | Environment launcher, common code, custom classes, and all tests | `bash shared/scripts/test_all.sh` |

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

The current environment already contains all runtime dependencies. For routine use, run the Bash entry points below directly. [run_vla.sh](../shared/scripts/run_vla.sh) selects `vla-supervised`, using `conda run -n vla-supervised` when it has not been manually activated; it does not use base or Windows Python. The current interpreter is `/home/administrator/miniconda3/envs/vla-supervised/bin/python`. If Conda is installed elsewhere, activate this environment first or set `CONDA_EXE=/path/to/conda/bin/conda`.

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
| matplotlib | 3.11.2 | Existing optional library; not required for BEV export |
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
bash tasks/sidewalk/scripts/setup_sidewalk.sh
bash tasks/nuimages/scripts/setup_nuimages.sh
```

[setup_sidewalk.sh](../tasks/sidewalk/scripts/setup_sidewalk.sh) adds PyArrow, Pillow, and NumPy according to [requirements.txt](../tasks/sidewalk/requirements.txt); [setup_nuimages.sh](../tasks/nuimages/scripts/setup_nuimages.sh) adds pycocotools, Pillow, and NumPy according to [requirements.txt](../tasks/nuimages/requirements.txt). Neither script installs SAM3, torch, or torchvision. nuScenes data reading and projection use NumPy/Pillow and do not require `nuscenes-devkit`. BEV rendering uses only NumPy/Pillow, without WebGL or matplotlib.

The model uses the local official `models/sam3/sam3.pt`. The model directory and SAM3 source directory are `/mnt/d/model_zoo/sam3` and `/mnt/d/wsl_workplace/sam3`, respectively: the weights directory does not automatically provide the `sam3` Python package. Model construction explicitly sets `load_from_HF=False`, so segmentation does not need to download weights.

### Check the environment and data

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# Print the actual interpreter, GPU availability, key libraries, and SAM3 source path; do not load the model.
bash shared/scripts/run_vla.sh - <<'PY'
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
bash tasks/sidewalk/scripts/segment_sidewalk.sh --inspect-data
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --inspect-data
bash tasks/nuimages/scripts/segment_nuimages.sh --inspect-data
```

If CUDA is unavailable during the environment check, first verify `nvidia-smi` and torch's CUDA wheel in WSL. All segmentation entry points also support `--device cpu --precision fp32`; CPU execution is slower than GPU execution.

## Command reference

All commands below assume you are in the project root. Use a **new or empty output directory** for each segmentation run or point-cloud export. The examples in the table limit the number of samples and do not start full-dataset inference. Append `--help` to a segmentation/export entry point for its complete parameter list.

| Purpose | Ready-to-run command | Output / default behavior |
|---|---|---|
| Sidewalk dataset-class segmentation and evaluation | `bash tasks/sidewalk/scripts/segment_sidewalk.sh --max-images 5 --output-dir outputs/sidewalk_demo` | Predictions, GT comparison visualizations, and pixel metrics; train by default |
| Sidewalk custom TXT classes | `bash tasks/sidewalk/scripts/segment_custom.sh --classes-file shared/config/classes.txt --max-images 5 --output-dir outputs/sidewalk_custom` | Predictions and visualizations, without GT metrics |
| nuScenes front camera, 16 classes | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --max-images 5 --output-dir outputs/nuscenes_demo` | mini_val / CAM_FRONT by default; lidarseg projected-point observation evaluation |
| nuScenes original 32 classes | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --taxonomy lidarseg32 --max-images 5 --output-dir outputs/nuscenes_raw32` | Preserves original lidarseg IDs |
| nuScenes six cameras + 2D BEV | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --bev-resolution 0.2 --output-dir outputs/nuscenes_six_demo` | Defaults to 6 camera images = 1 LiDAR frame; automatic fusion, 2D BEV export, and unique-point metrics |
| nuScenes six cameras over 10 frames | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --max-images 60 --bev-resolution 0.2 --output-dir outputs/nuscenes_six_10frames` | 60 images = 10 frames, fused independently per frame |
| nuScenes custom TXT classes | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_custom.sh --classes-file shared/config/classes.txt --max-images 5 --output-dir outputs/nuscenes_custom` | Image segmentation and visualization, without class GT metrics |
| nuScenes six cameras with custom TXT classes + 2D BEV | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --mode custom --classes-file shared/config/classes.txt --bev-resolution 0.2 --output-dir outputs/nuscenes_six_custom` | Prediction/coverage BEV and full NPZ, without lidarseg semantic metrics |
| Color a point cloud using existing nuScenes segmentation results | `bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --bev-resolution 0.2 --output-dir outputs/nuscenes_demo_cloud` | No SAM3 rerun; 2D BEV PNG/raster/metadata, full NPZ, and unique-point evaluation in dataset mode |
| Export only 1 point-cloud frame from existing results | `bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --max-frames 1 --bev-resolution 0.2 --output-dir outputs/nuscenes_first_cloud` | Count is in LiDAR keyframes |
| nuImages class segmentation, GT comparison, and evaluation | `bash tasks/nuimages/scripts/segment_nuimages.sh --max-images 5 --output-dir outputs/nuimages_demo` | v1.0-mini by default, 24 non-ego classes; six-panel comparison and pixel metrics |
| nuImages front-view filtering only | `bash tasks/nuimages/scripts/segment_nuimages.sh --cameras CAM_FRONT --max-images 5 --output-dir outputs/nuimages_front` | Selects only annotated key-camera images |
| nuImages prompting and evaluation including ego | `bash tasks/nuimages/scripts/segment_nuimages.sh --include-ego --max-images 5 --output-dir outputs/nuimages_with_ego` | Includes ID31, for 25 prompted classes |
| nuImages custom TXT classes | `bash tasks/nuimages/scripts/segment_nuimages_custom.sh --classes-file shared/config/classes.txt --max-images 5 --output-dir outputs/nuimages_custom` | Image segmentation and visualization, without GT metrics |
| Create/check the nuImages symlink | `bash tasks/nuimages/scripts/link_nuimages.sh` | Points to `/mnt/d/datasets/nuimages`; does not overwrite an existing different target |
| Add Sidewalk data-reading dependencies | `bash tasks/sidewalk/scripts/setup_sidewalk.sh` | Always installs into vla-supervised |
| Add nuImages mask-decoding dependencies | `bash tasks/nuimages/scripts/setup_nuimages.sh` | Always installs into vla-supervised |
| All regression tests | `bash shared/scripts/test_all.sh` | Runs every task and shared regression test |
| nuScenes data/export pipeline tests | `bash tasks/nuscenes_sam3/scripts/test_nuscenes.sh` | All nuScenes SAM3 task tests, including BEV, fusion, and viewer |
| nuImages data/pipeline tests | `bash tasks/nuimages/scripts/test_nuimages.sh` | All nuImages task tests |
| EFFOcc Fusion-R18 mini inference | `bash tasks/effocc/scripts/run_effocc_mini.sh --max-samples 1 --output-dir outputs/effocc_demo` | Six cameras plus LiDAR/sweeps; original 200x200x16 occupancy and 2D BEV |
| Same-frame SAM3 / EFFOcc comparison | `bash tasks/effocc/scripts/compare_effocc_sam3.sh --input-dir outputs/effocc_mini_with_cameras --max-samples 1 --output-dir outputs/effocc_sam3_demo` | Original, SAM3 and occupancy overlays for all six exact camera frames |
| EFFOcc occupancy projected into six cameras | `bash tasks/effocc/scripts/project_effocc_cameras.sh --input-dir outputs/effocc_mini_verified --max-samples 1 --output-dir outputs/effocc_camera_demo` | Reuse saved occupancy; original-size overlays, rendered classes/depth and six-camera sheets |
| EFFOcc input inspection | `bash tasks/effocc/scripts/run_effocc_mini.sh --inspect-data` | Validate mini input arrays without running the model |
| EFFOcc adapter tests | `bash tasks/effocc/scripts/test_effocc.sh` | Sparse operator, preprocessing, pooling and output tests |
| Execute Python commands in the designated environment | `bash shared/scripts/run_vla.sh -m pip list` | General environment entry point; accepts any Python arguments |

### Counts, parameters, and viewing results

| Parameter / default | Sidewalk | General nuScenes entry point | Six-camera nuScenes entry point | nuImages |
|---|---|---|---|---|
| Default number processed | All remaining images | 10 camera images | 6 camera images = 1 frame | 10 annotated images |
| Unit of `--max-images N` | Images | Camera images | Camera images; multiples of 6 recommended | Annotated images after filtering |
| `--max-images 0` | All remaining images | All selected views | All selected six-camera views | All remaining keyframes |
| Data selection | `--split train/validation/test` | `--split all/mini_train/mini_val` | Same as nuScenes | `--version v1.0-mini` / `v1.0-train` / `v1.0-val` / `v1.0-test` |
| Class mode | Dataset / custom | `lidarseg16/lidarseg32` / custom | Same as nuScenes | Dataset / custom; can include ego |

Segmentation entry points for all three datasets support `--start-index`, `--classes-file`, `--score-threshold`, `--mask-threshold`, `--device`, `--precision`, `--alpha`, `--vis-every`, `--no-vis`, `--inspect-data`, and `--output-dir`. Both thresholds default to 0.5, and the default execution mode is `cuda/bf16`. Override model and data paths with `--checkpoint` and `--dataset-dir`. The selected split/version must actually exist; locally, Sidewalk has only train, and nuScenes and nuImages have only mini.

nuScenes uses `--cameras`, `--scenes`, `--min-depth`, and `--export-pointcloud`. In nuImages, `--cameras` filters annotated key cameras; it does not mean that each sample has six annotations. Standalone BEV export supports `--max-frames`, `--bev-resolution` (default 0.2 meters/pixel), `--bev-density-cap` (default 32 winning-class points at the darkest shade), and `--no-preview`. Segmentation accepts both BEV parameters, which affect display only; NPZ and point-level metrics retain all points.

- 2D comparison images: `visualizations/` in each segmentation output; raw class maps are in `predictions/`.
- Image metrics / camera projected-point observation metrics: `metrics.json` and `metrics.csv` at the segmentation output root.
- Six-camera BEV: `pointcloud_results/pointclouds/<scene_frame_sample>/` contains 2D PNGs, raster arrays, metadata, and complete original-point NPZ.
- Fused unique-point metrics: `evaluation/metrics.json` in the point-cloud output, reporting covered_points, full_scan, and coverage separately. Their scope differs from the projected-observation metrics at the segmentation output root.
- Custom TXT mode does not output semantic GT evaluation; custom nuScenes point clouds still report geometric coverage.

Class mappings, output fields, and evaluation details for each mode are described in the following sections.

## Project layout and data paths

Only **WSL + Conda `vla-supervised`** is supported. Launchers select this environment rather than base or Windows Python. Each task owns its Python implementation, Bash launchers, tests, and task-specific class configuration or requirements. Common runtime, SAM3 support, metrics, and nuScenes data utilities live in [shared/python/](../shared/python/).

```text
tasks/
  sidewalk/
    python/                   # sam3_sidewalk.py
    scripts/                  # segment_sidewalk.sh, segment_custom.sh, setup_sidewalk.sh, test_sidewalk.sh
    config/sidewalk_classes.txt
    tests/
    requirements.txt
  nuimages/
    python/                   # sam3_nuimages.py, nuimages_data.py
    scripts/                  # link_nuimages.sh, segment_nuimages.sh, segment_nuimages_custom.sh,
                              # setup_nuimages.sh, test_nuimages.sh
    config/nuimages_classes.txt
    tests/
    requirements.txt
  nuscenes_sam3/
    python/                   # sam3_nuscenes.py, nuscenes_pointcloud.py, pointcloud_semantics.py,
                              # pointcloud_bev.py, pointcloud_viewer.py
    scripts/                  # segment_nuscenes.sh, segment_nuscenes_custom.sh, segment_nuscenes_6cams.sh,
                              # color_nuscenes_pointcloud.sh, test_nuscenes.sh
    config/                   # nuscenes_classes.txt, nuscenes_lidarseg32_classes.txt
    tests/
  effocc/
    python/                   # effocc_mini.py, effocc_data.py, effocc_model.py, effocc_sparse.py,
                              # effocc_cameras.py, effocc_projection.py, effocc_sam3_compare.py
    scripts/                  # run_effocc_mini.sh, compare_effocc_sam3.sh, project_effocc_cameras.sh,
                              # test_effocc.sh
    tests/
    requirements.txt
shared/
  python/                     # runtime.py, sam3.py, segmentation_metrics.py, nuscenes_data.py
  config/classes.txt          # Custom TXT classes for all tasks
  scripts/                    # run_vla.sh, test_all.sh
python/                       # Legacy Python forwarding entry points
scripts/                      # Legacy Bash forwarding entry points, including test_all.sh
config/                       # Legacy class-configuration copies
tests/                        # Legacy test imports forwarding to the new task/shared locations
requirements-*.txt            # Legacy aliases: -r tasks/<task>/requirements.txt
datasets/                     # Existing data and symlinks
models/                       # Existing model sources and SAM3 weights symlink
assets/                       # Existing checkpoints
outputs/                      # Existing and new results; default run names are preserved
```

Run canonical Python modules **from the project root** in the activated environment:

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad
python -m tasks.sidewalk.python.sam3_sidewalk --help
python -m tasks.nuimages.python.sam3_nuimages --help
python -m tasks.nuscenes_sam3.python.sam3_nuscenes --help
python -m tasks.effocc.python.effocc_mini --help

# Select the designated environment without manual activation.
bash shared/scripts/run_vla.sh -m tasks.nuscenes_sam3.python.sam3_nuscenes --help
```

Task Bash launchers automatically change to the project root, then call [shared/scripts/run_vla.sh](../shared/scripts/run_vla.sh) with `-m tasks.<task>.python.<module>`. Their filenames and CLI parameters are preserved. Shared [runtime.py](../shared/python/runtime.py) and [sam3.py](../shared/python/sam3.py) provide common execution/model support; [segmentation_metrics.py](../shared/python/segmentation_metrics.py) provides metrics, and [nuscenes_data.py](../shared/python/nuscenes_data.py) supplies nuScenes data/projection utilities to SAM3 and EFFOcc.

[Sidewalk tests](../tasks/sidewalk/scripts/test_sidewalk.sh) run only Sidewalk tests; [nuImages tests](../tasks/nuimages/scripts/test_nuimages.sh) run only nuImages tests; [nuScenes SAM3 tests](../tasks/nuscenes_sam3/scripts/test_nuscenes.sh) include every test in that task, including BEV, fusion, and viewer coverage; [EFFOcc tests](../tasks/effocc/scripts/test_effocc.sh) cover its adapter. Use [shared/scripts/test_all.sh](../shared/scripts/test_all.sh) for all tests. The legacy [scripts/test_sidewalk.sh](../scripts/test_sidewalk.sh) still runs the entire suite, and [scripts/test_all.sh](../scripts/test_all.sh) is an additional compatibility entry point. Root test modules retain compatibility imports of their actual task/shared implementations.

The original top-level Python and Bash paths remain forwarding entry points; top-level class TXT files remain compatibility copies. The three legacy requirements files delegate with `-r tasks/<task>/requirements.txt`; prefer the task requirements and class configurations shown above, and [shared/config/classes.txt](../shared/config/classes.txt) for all custom-class modes. The EFFOcc/SAM3 comparison reuses the [nuScenes 16-class configuration](../tasks/nuscenes_sam3/config/nuscenes_classes.txt) by default. This reorganization changes the import layout and entry-point locations; it requires no environment reinstallation. Data, model sources, checkpoints, and outputs stay in place, model weights are unchanged, and old/new output default names are preserved. The inference and validation notes below describe existing runs; no new inference results are claimed for this layout change.

The project uses these existing WSL symlinks:

- `models/sam3` → `/mnt/d/model_zoo/sam3`, using the official `sam3.pt`.
- `datasets/sidewalk-semantic` → `/mnt/d/datasets/sidewalk-semantic`, reading `data/train-*.parquet` and `id2label.json` directly.
- `datasets/nuscenes` → `/mnt/d/datasets/nuscenes`, using `v1.0-mini/`, `samples/`, and `lidarseg/v1.0-mini/`.
- `datasets/nuimages` → `/mnt/d/datasets/nuimages`, using 2D annotated images from `v1.0-mini/` and `samples/`.

See [Environment setup](#environment-setup) above for activating the current environment, adding dependencies, and creating a new environment. The sections below describe class configuration, outputs, and evaluation for each dataset.

## 1. Sidewalk Semantic classes and evaluation

First check the output with a few images:

```bash
bash tasks/sidewalk/scripts/segment_sidewalk.sh \
  --max-images 5 \
  --output-dir outputs/sidewalk_demo
```

Evaluate all 1000 training images:

```bash
bash tasks/sidewalk/scripts/segment_sidewalk.sh \
  --output-dir outputs/sidewalk_full
```

By default, the original 34 nonzero classes are read, preserving dataset IDs 1–34; ID 0 is `unlabeled`. [sidewalk_classes.txt](../tasks/sidewalk/config/sidewalk_classes.txt) maps dataset label names such as `flat-sidewalk` to English prompts such as `sidewalk` that better suit SAM3. You can modify prompts to the right of `=` and add aliases with `|`; the left side must retain the complete original class name. You can also specify a complete replacement mapping with `--classes-file tasks/sidewalk/config/your_sidewalk_prompts.txt`.

The original classes include `void-ground`, `void-dynamic`, `void-static`, and `void-unclear`. These labels lack precise natural-language visual concepts, so the configuration uses approximate prompts, while evaluation still scores the original IDs. This is zero-shot semantic segmentation evaluation using SAM3 text prompts, without a trained semantic classification head. Prompts, confidence thresholds, and mask-conflict handling all affect the metrics.

Only the `train` split is available locally. The script also accepts `--split validation/test`, but you must provide the corresponding Parquet files in the same format. When running on a subset, metrics cover only the images actually processed and do not represent the entire dataset.

## 2. Custom TXT classes

Edit [classes.txt](../shared/config/classes.txt), with one English class per line:

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
bash tasks/sidewalk/scripts/segment_custom.sh \
  --classes-file shared/config/classes.txt \
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
bash tasks/sidewalk/scripts/segment_sidewalk.sh --inspect-data

# Process 10 images from global index 100, generating fewer visualizations
bash tasks/sidewalk/scripts/segment_sidewalk.sh \
  --start-index 100 --max-images 10 --vis-every 5 \
  --score-threshold 0.5 --mask-threshold 0.5 \
  --output-dir outputs/sidewalk_subset

# Show all parameters
bash tasks/sidewalk/scripts/segment_sidewalk.sh --help

# All regression tests, including class configuration, merging, data reading, visualization, and metrics
bash shared/scripts/test_all.sh
```

The default `--device cuda --precision bf16` suits this machine's RTX 5070 Ti. `--precision fp32/fp16` is supported; CPU mode requires both `--device cpu --precision fp32`. `--no-vis` saves only class-ID maps, instance information, and evaluation results. `--max-images 0` processes every image from the start index to the end.

## 3. nuScenes mini: camera segmentation and lidarseg projected-point evaluation

Run only in WSL's `vla-supervised` environment, reusing local SAM3. JSON reading and point-cloud projection use NumPy/Pillow, without installing `nuscenes-devkit` or changing the existing environment.

The default uses `CAM_FRONT` from `mini_val`, processes **at most 10 images**, and uses the official 32→16 class mapping. You can begin by limiting it to 1–5 images:

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --max-images 5 \
  --output-dir outputs/nuscenes_demo
```

Select one scene from the full mini dataset and multiple cameras:

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --split all --scenes scene-0061 \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
  --max-images 6 \
  --output-dir outputs/nuscenes_scene0061
```

`--split` supports `mini_train`, `mini_val`, and `all`. Scene filtering is intersected with the split. The six available cameras are `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, and `CAM_BACK_RIGHT`. Only keyframes are used, associated with `LIDAR_TOP` and lidarseg labels from the same sample; sweeps are not used. Enumeration follows scene name, sample timestamp, and the specified camera order.

The local mini dataset contains 404 samples, or 2,424 images across all six cameras; mini_train has 323 samples and mini_val has 81. **`--max-images` and `--start-index` count camera images in the selected list**. For example, with six cameras, `--max-images 6` processes all six views of one sample. Only `--max-images 0` processes every selected view from the start index to the end. Each run requires a new or empty output directory.

### Class modes

- Default `--taxonomy lidarseg16`: 16 valid classes, IDs 1–16; ID 0 means ignore/uncovered. Uses [nuscenes_classes.txt](../tasks/nuscenes_sam3/config/nuscenes_classes.txt); retain the class names on the left, and modify prompts or add `|` aliases on the right. The classes are barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain, manmade, and vegetation.
- `--taxonomy lidarseg32`: preserves original label IDs 0–31; 0=noise is ignored, and all IDs 1–31 are prompted and evaluated. Uses [nuscenes_lidarseg32_classes.txt](../tasks/nuscenes_sam3/config/nuscenes_lidarseg32_classes.txt). Text prompts for pedestrian subcategories and `vehicle.ego`, among others, have semantic ambiguities. This mode is an experiment with the original classes; the official benchmark generally uses the merged 16 classes.
- `--classes-file` can replace the complete prompt configuration for that taxonomy; class names and IDs stay the same.

```bash
# Original lidarseg 32 classes
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --taxonomy lidarseg32 --max-images 5 \
  --output-dir outputs/nuscenes_raw32

# Custom TXT classes: segmentation and visualization only, without lidarseg metrics
bash tasks/nuscenes_sam3/scripts/segment_nuscenes_custom.sh \
  --classes-file shared/config/classes.txt --max-images 5 \
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
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --inspect-data

# Show all parameters
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --help

# All nuScenes SAM3 tests, including projection, metrics, BEV, fusion, and viewer
bash tasks/nuscenes_sam3/scripts/test_nuscenes.sh
```

## 4. Color original LiDAR points and export 2D BEV

Pixel predictions are assigned back to the **original point indices** using the same calibrated projection, then fused across completed camera views. The colored points are flattened onto an XY bird's-eye-view (BEV) raster. Coordinates remain in the native `LIDAR_TOP` sensor frame, in meters; **+x points right and +y points up**, without assuming the vehicle front faces upward. Original point arrays and point-level evaluation retain the full scan.

### Export existing segmentation results without rerunning inference

```bash
bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh \
  --input-dir outputs/nuscenes_demo \
  --bev-resolution 0.2 \
  --output-dir outputs/nuscenes_demo_cloud
```

The input is a completed nuScenes segmentation output directory; the script reads `samples.jsonl`, `predictions/`, and the configuration. Images recorded as completed in a failed/interrupted run can also be exported, with the source status preserved in the output. Do not export while the source segmentation task is still running. Custom-class mode can also export predicted point clouds and coverage, but does not compute class metrics against lidarseg.

`--max-frames 1` exports just one LiDAR keyframe; the default 0 exports all completed point-cloud frames in the input. `--dataset-dir` can override the data root in the source configuration. Use a new or empty output directory to avoid mixing results from different inference runs.

### Run six-camera segmentation, coloring, and evaluation together

The dedicated six-view entry point processes one keyframe (6 camera images) by default and automatically exports 2D BEV, full NPZ, and metrics:

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --bev-resolution 0.2 --output-dir outputs/nuscenes_six_demo
```

To process 10 frames, specify `--max-images 60`. Multiples of 6 are recommended for `--max-images` and `--start-index` to avoid having only some views for the first or last frame. You can also use the general entry point:

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \
  --max-images 6 \
  --export-pointcloud \
  --bev-resolution 0.2 \
  --output-dir outputs/nuscenes_one_scan
```

Here, 6 images correspond to one sample's six cameras and are fused into one colored point-cloud frame. The default is `--split mini_val --taxonomy lidarseg16`; the original `lidarseg32` or custom-class entry point can also be used. Add `--export-pointcloud` to the existing segmentation command to export automatically to `<output-dir>/pointcloud_results/`. Existing single-camera results cover only that camera's region; exporting a point cloud does not automatically run the other cameras. If `--start-index`/`--max-images` truncates the camera list, only the views actually completed for that frame are fused, and the camera list is recorded in the output.

When the same original point appears in multiple cameras, fusion uses a **majority vote among nonzero classes**. Class 0 means segmentation did not assign a class and is treated as an abstention, but the point still counts as geometrically covered. Ties favor the earlier camera in the input list. Each point is evaluated once after fusion to avoid counting camera-overlap regions repeatedly. Fusion does not read GT or use ground truth to select predictions.

### BEV images and original point arrays

```text
<pointcloud-output>/
  run_config.json / frames.jsonl / classes.json
  evaluation/
    metrics.json / metrics.csv / coverage.csv
    confusion_matrix_covered.csv / confusion_matrix_full_scan.csv
  pointclouds/<scene_frame_sample>/
    points.npz                # Full original point order, labels, coverage and fusion sources
    bev_raster.npz            # Majority labels, total/winning/covered counts and sparse class histograms
    bev_metadata.json         # Resolution, XY bounds, count rules and density scale
    bev_prediction.png        # Majority predicted class, shaded by winning-class point count
    bev_ground_truth.png      # Independently voted GT class, shaded by its winning count
    bev_errors.png            # Cell-majority comparison; not the original point-level metric
    bev_coverage.png          # Any-covered cells, shaded by covered point count
    bev_density.png           # Total points per cell, grayscale density
    bev_comparison.png        # 2x2 with GT and density legends; two panels for custom mode
    summary.json / evaluation/
```

Automatic export places these files under `<output-dir>/pointcloud_results/`; standalone export uses its output root. The default is a **fixed 100 m x 100 m semantic grid at 0.2 m x 0.2 m per cell: 500 x 500 pixels**. The existing native LiDAR coordinates and semantic classes are retained: X/Y bounds are `[-50, 50)` meters, **+x right, +y up**. `--bev-roi-size 100` controls the square side length; `--bev-resolution 0.2` controls cell size. The ratio must be an integer. `--bev-roi-size 0` restores the previous automatic full-point extent. Only points inside the ROI vote in the displayed grid; outside points remain in the full `points.npz` and point-level metrics. Metadata reports `roi_input_point_count`, ROI `point_count`, `roi_outside_point_count` and `grid_mode`. No driveable/obstacle/unknown reclassification is performed.

Every original unique point votes once for its class in its XY cell. **The class with the most points becomes the cell class; ties choose the lowest class ID.** Prediction and GT are voted independently on the same grid. Class 0 participates as unassigned/ignored; it is not silently dropped to favor a nonzero prediction. Z and point order do not affect cell labels. Camera fusion still precedes this vote, so a point seen by several cameras is not counted several times.

**The more points supporting the winning class, the darker its color. Empty cells are pure white.** Color is blended from white toward the class palette using an absolute logarithmic count scale: `alpha = 0.25 + 0.75 * log1p(min(n, cap)) / log1p(cap)` for `n > 0`, and `RGB = round(255 * (1 - alpha) + base_RGB * alpha)`. The default `--bev-density-cap 32` reaches the darkest palette color at 32 or more supporting points. This is a count, not confidence or probability; no per-frame normalization is used. Prediction and GT use their own winning-class counts. The comparison includes 1/2/4/8/16/32-point color examples.

Cells with unassigned class 0 are gray, with the same density shading. Coverage means any original point in the cell is camera-covered; its blue shade uses `covered_count`. `bev_density.png` uses total cell point counts. Error colors compare **cell majorities**, only for covered cells with nonzero GT majority: green means equal classes, red unequal classes, and gray ignored cells. These colors are distinct from the full original-point metrics reported in the header.

`bev_raster.npz` contains HxW `occupied/covered` booleans, `prediction` and optional `target` uint8 labels (255 for empty cells), and uint32 `point_count`, `prediction_count`, optional `target_count`, and `covered_count`. Counts are zero in empty cells. Sparse arrays `prediction_vote_cells/prediction_vote_classes/prediction_vote_counts` retain each cell/class histogram; optional `target_vote_*` records GT votes. Cell IDs are row-major flat indices. The former single-point `point_index` field is removed because a cell now aggregates all points.

`points.npz` retains the full original order and `xyz/intensity/ring/raw_labels/point_indices/prediction/covered/observation_count/winning_votes/source_camera/camera_names`, plus `target` in dataset-class mode. **All original points remain in point-level evaluation.** Resolution and density color changes do not alter those metrics; no BEV pixel metric is added. No cells are interpolated or filled.

For one frame with all six cameras:

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh \
  --bev-roi-size 100 --bev-resolution 0.2 --bev-density-cap 32 \
  --output-dir outputs/nuscenes_single_frame_grid
```

To reuse an already segmented six-camera frame without rerunning SAM3:

```bash
bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh \
  --input-dir outputs/smoke_nuscenes_six_cloud --max-frames 1 \
  --bev-roi-size 100 --bev-resolution 0.2 --bev-density-cap 32 \
  --output-dir outputs/nuscenes_single_frame_majority
```

### Point-cloud evaluation conventions

- `covered_points`: evaluates only unique points with geometric projection coverage from at least one processed camera, ignoring GT0. A covered point with prediction0 is still an error; selecting only successfully colored points cannot inflate the metrics.
- `full_scan`: includes every nonzero GT point in the entire frame, with prediction0 on uncovered points counted as an error. It reflects the ability to assign classes throughout the full frame; this score is typically much lower for a single camera.
- `coverage`: reports total points, covered points, points with actual nonzero predictions, valid GT coverage, and per-class coverage. Metrics are available per frame and across all exported frames; the GT merging policy still follows the source configuration's 16/32-class mode.
- Both scopes output point accuracy, mIoU, mean accuracy, frequency-weighted IoU, macro Dice, and per-class precision/recall/IoU/TP/FP/FN. A prediction of 0 creates an FN for the corresponding GT class; GT0 is entirely excluded from macro metrics and the confusion matrix. The same point across multiple cameras is counted once, and point indices are independent for each frame.
- Calibration, occlusion across pixels, and timing differences for dynamic objects still introduce errors. Point-cloud results come from camera predictions; they are not outputs of a trained 3D semantic segmentation network or official nuScenes benchmark submission scores.

`--no-preview` disables all BEV PNGs but retains point/raster NPZ, metadata, summaries, and metrics. `--export-pointcloud --no-vis` has the same effect. Migration: `--html-max-points` and `--no-html` are removed; no HTML, PLY, or 3D previews are generated. Historical outputs are not deleted. Point-cloud metrics are in the point-cloud output's `evaluation/` directory; `metrics.json` at the original segmentation output root still counts projected-point observations per camera. The two evaluation scopes differ.

## 5. nuImages: image segmentation, ground-truth comparison, evaluation, and visualization

The local data at `D:\datasets\nuimages` is **v1.0-mini, with 50 annotated keyframes**. The WSL symlink `datasets/nuimages -> /mnt/d/datasets/nuimages` has been created. You can rerun the following command to check/create the link; an existing different link or directory will not be overwritten:

```bash
bash tasks/nuimages/scripts/link_nuimages.sh
```

The current WSL `vla-supervised` environment already includes SAM3, PyTorch, Pillow, NumPy, and `pycocotools` for RLE decoding; no additional devkit is needed. If another environment with the same name lacks the decoding dependencies, run `bash tasks/nuimages/scripts/setup_nuimages.sh`. The additional dependencies are recorded in [requirements.txt](../tasks/nuimages/requirements.txt).

### Dataset-class mode

```bash
bash tasks/nuimages/scripts/segment_nuimages.sh \
  --max-images 5 \
  --output-dir outputs/nuimages_demo
```

The default is `--version v1.0-mini`, processing at most 10 images and selecting **annotated key cameras** from all camera channels. The local mini keyframes are distributed as follows: front 8, front-left 5, front-right 7, back 11, back-left 10, and back-right 9. Each nuImages sample provides one annotated key camera and unannotated sweeps before and after it from that camera, unlike the six synchronized cameras in a nuScenes sample. The script does not treat sweeps as samples with GT or invent shared annotations for six cameras.

Filter cameras, for example to compare only front views:

```bash
bash tasks/nuimages/scripts/segment_nuimages.sh \
  --cameras CAM_FRONT --max-images 5 \
  --output-dir outputs/nuimages_front
```

`--cameras` accepts multiple names from the six camera channels; the default applies no filter. Images are sorted by image timestamp and sample token; `--start-index` and `--max-images` refer to the filtered keyframe list. The default processes only 10 images; only `--max-images 0` processes every selected image from the start index to the end. The output directory must be new or empty.

`--version v1.0-train/v1.0-val` supports data in the same format, but you must supply the corresponding metadata and images; only mini is currently available locally. `v1.0-test` has no available ground truth, so the script allows inference only in custom-class mode.

### Classes and ground-truth composition

Uses [nuimages_classes.txt](../tasks/nuimages/config/nuimages_classes.txt): retain original class names on the left, and modify English prompts or add aliases with `|` on the right. You can also specify a complete replacement prompt file with `--classes-file`. Official IDs are preserved:

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
bash tasks/nuimages/scripts/segment_nuimages_custom.sh \
  --classes-file shared/config/classes.txt --max-images 5 \
  --output-dir outputs/nuimages_custom
```

Classes receive IDs 1..N in TXT order, with 0 meaning unassigned, as in the other custom entry points. This mode reads only image metadata and does not require annotation tables. It outputs predictions, color maps, overlays, and three-panel comparisons, without GT/error maps or semantic evaluation, because custom classes may not correspond to the annotations.

```bash
# Inspect image selection, annotation decoding, class IDs, and valid-pixel counts, without loading SAM3
bash tasks/nuimages/scripts/segment_nuimages.sh --inspect-data

# Show parameters or run nuImages regression tests
bash tasks/nuimages/scripts/segment_nuimages.sh --help
bash tasks/nuimages/scripts/test_nuimages.sh
```

## 6. EFFOcc Fusion-R18 on nuScenes mini

Run the checkpoint at `assets/checkpoints/effocc_fusion_r18.pth` using the existing **WSL `vla-supervised`** environment. The Bash entry point defaults to **one LiDAR sample and all six cameras**, with the current LiDAR keyframe plus up to nine preceding sweeps. `--max-samples` counts samples, not camera images. No dataset modification, BEVDet pickle generation, model download, or environment downgrade is needed for this adapter.

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# Check image, calibration, sweep, point-feature and depth-map inputs only.
bash tasks/effocc/scripts/run_effocc_mini.sh --inspect-data

# Run one six-camera sample with the supplied checkpoint.
bash tasks/effocc/scripts/run_effocc_mini.sh \
  --max-samples 1 \
  --output-dir outputs/effocc_demo

# The next sample has real preceding sweeps rather than empty-history padding.
bash tasks/effocc/scripts/run_effocc_mini.sh \
  --start-index 1 --max-samples 1 \
  --output-dir outputs/effocc_history_demo

# Select a scene from the complete mini split.
bash tasks/effocc/scripts/run_effocc_mini.sh \
  --split all --scenes scene-0061 --max-samples 1 \
  --output-dir outputs/effocc_scene0061

# Show options and run the adapter tests.
bash tasks/effocc/scripts/run_effocc_mini.sh --help
bash tasks/effocc/scripts/test_effocc.sh
```

Default data is `datasets/nuscenes`, split `mini_val`. Override paths with `--dataset-dir` and `--checkpoint`; use a new or empty `--output-dir`. `--max-samples 0` processes all remaining selected samples. `--device cpu` is supported by the adapted operations for low-speed checks; the real examples were verified on CUDA. Unlike SAM3, this entry point uses float32, with TF32 disabled while validating the adaptation. It does not accept text classes or SAM3's `--max-images/--precision` options. This is a **camera + LiDAR fusion model**, not a camera-only model: it consumes image features, sparse LiDAR features and LiDAR-derived image depth. By default, each new inference run also renders occupancy into all six original camera images; use `--no-camera-projection` to skip rendering or `--projection-alpha 0.45` to adjust overlay opacity.

### Checkpoint and native grid

The supplied weights strictly match the Occ3D configuration [flashocc_fusion_r18_base_100%_seqs.py](../models/EFFOcc/configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py): 387 state tensors, with no missing or unexpected keys. This is **18-class Occ3D**, not the 17-class OpenOccupancy configuration.

- Six images are resized/cropped to 256x704.
- Image BEV features are 1x64x200x200, and LiDAR BEV features are 1x256x200x200 before fusion.
- The prediction is uint8 **200x200x16**, in X,Y,Z order, with 0.4m output voxels.
- Native range is X/Y `[-40,40)`m and Z `[-1,5.4)`m. Class 17 is free; classes 0..16 are the original semantic classes.
- The LiDAR encoder's input voxel size is 0.05x0.05x0.16m. It is distinct from the 0.4m output voxel grid.
- Output BEV is **200x200** at the native 80mx80m scale. The prior SAM3 500x500 grid is a separate rasterization setting and must not be substituted into this trained network.

```text
outputs/<effocc_run>/
  run_config.json               # Environment, checkpoint SHA256, strict-load status and timing
  classes.json                  # Original 18 class names
  <scene_index_sample>/
    occupancy.npz               # semantics[X,Y,Z], native range, voxel size and class names
    bev_class_ids.png           # 200x200 class IDs of the highest non-free voxel per XY column
    bev_semantics.png           # Native 200x200 semantic color map; free columns are white
    bev_preview.png             # Enlarged 2D map with class legend and grid description
    sample.json                 # Input paths, sweep selection, normalization, shapes and finite-logit checks
```

### Runtime compatibility and validation scope

The original [EFFOcc repository](../models/EFFOcc/README.md) imports legacy MMCV 1.5.2..1.7.0, mmdet2/mmseg0 APIs, compiled sparse operations and its custom CUDA BEVPoolv2. Those are absent from the current Python3.12/torch2.14/cu132 environment, and no `nvcc` is installed. The original `tools/test.py` therefore does not run unchanged here.

The project adds an **inference-only PyTorch adaptation** in [effocc_model.py](../tasks/effocc/python/effocc_model.py), [effocc_sparse.py](../tasks/effocc/python/effocc_sparse.py) and [effocc_data.py](../tasks/effocc/python/effocc_data.py). It preserves the image, sparse LiDAR, depth, fusion and occupancy branches and loads every checkpoint tensor strictly. Sparse submanifold/strided convolutions use active-coordinate matching and matrix multiplication; BEVPoolv2 uses chunked weighted `index_add_`. The original `models/EFFOcc` checkout is not modified. Architecture/pipeline sources retain their EFFOcc/OpenMMLab/BEVDet attribution and Apache-2.0 license context.

Runtime dependencies are existing torch, torchvision, NumPy, Pillow and OpenCV (`cv2`); [requirements.txt](../tasks/effocc/requirements.txt) lists the extra non-torch dependencies for another environment. No packages were installed or downgraded for the local run. A fresh environment needs matching CUDA torch/torchvision first. If installing listed requirements together with SAM3, honor SAM3's declared NumPy constraints rather than treating this file as a lock file.

Input handling follows source behavior, including Pillow's default BICUBIC crop/resize, PIL RGB reversed to BGR by the upstream `mmlabNormalize(to_rgb=True)` operation, fifth point feature as time lag rather than ring index, and the source's different CAM_FRONT point-reference / CAM_FRONT_LEFT image-reference conventions. The adapter selects the nearest nine previous sweeps deterministically, while the original configuration can choose randomly; nearest-pixel depth reduction uses a stable minimum rather than reproducing float32 rank-sort quantization. These differences are recorded in `sample.json`.

**Verified:** two actual mini samples, including an empty-history padded first frame and a next frame with nine real sweeps; 387/387 tensor loading; finite full-branch logits; and 53 operator/input/output/rendering tests including CPU/CUDA sparse-convolution comparisons against dense reference operations. These checks establish that inference runs. **Full-model output equivalence with the original MMCV backend and the paper's accuracy have not been established.**

The local lidarseg point labels are not Occ3D dense voxel GT. Predictions and visualizations require no occupancy labels, but this command intentionally produces **no occupancy mIoU**. Formal Occ3D evaluation requires the matching dense `labels.npz` annotations and their masks; do not interpret the preview or sparse lidarseg checks as benchmark scores.

### Occupancy projected into all six original cameras

New inference runs create a `camera_projections/` directory inside each sample output by default. You can also render an existing prediction without rerunning the network:

```bash
bash tasks/effocc/scripts/project_effocc_cameras.sh \
  --input-dir outputs/effocc_mini_verified \
  --max-samples 1 --alpha 0.45 \
  --output-dir outputs/effocc_camera_demo
```

`--input-dir` accepts an EFFOcc run root or one sample directory containing `occupancy.npz` and `sample.json`. `--max-samples` defaults to 1; 0 renders all saved samples. Override `--dataset-dir` when the original nuScenes root moves. Standalone projection uses `--device cuda` by default; CPU is supported but slower. Use a new/empty output directory.

```text
<sample-or-projection-output>/camera_projections/  # Automatic inference output
  CAM_FRONT_LEFT_original.jpg                    # Same pattern for all six camera channels
  CAM_FRONT_LEFT_overlay.jpg                     # Original-size RGB image with semantic occupancy overlay
  CAM_FRONT_LEFT_semantics.png                   # Rendered surface colors; no hit white
  CAM_FRONT_LEFT_class_ids.png                   # uint8 0..16; 255 for no hit
  CAM_FRONT_LEFT_projection.npz                  # labels, camera-Z depth, voxel_index, camera_to_grid, intrinsic
  six_cameras_original.jpg                       # 3x2 original-image overview
  six_cameras_overlay.jpg                        # 3x2 occupancy overlay overview with class legend
  six_cameras_semantics.jpg                      # 3x2 rendered semantic surfaces
  projection_config.json                         # Grid reference, camera tokens, transforms, ray policy and status
```

Standalone output places these files under `<output-dir>/<sample-name>/`. Individual images retain the original camera size (1600x900 locally); contact sheets are downscaled for viewing. Rendering uses every original-image pixel center, `K^-1 [u+0.5,v+0.5,1]`, and exact voxel traversal to select the **first occupied voxel intersected by the ray**. Class17/free is transparent; classes0..16 are opaque (others0 is gray). No-hit pixels preserve the original image in overlays. This projects actual voxel volumes with depth occlusion, not fixed-radius dots or an image-space dilation. Native 0.4m voxel boundaries remain visible.

The default grid frame is **ego at CAM_FRONT_LEFT's timestamp**, matching the adapter's image key-ego convention recorded in `sample.json`. Each camera uses its own ego pose and sensor calibration, with `camera_to_grid = inverse(key_ego_to_global) @ camera_ego_to_global @ camera_sensor_to_ego`. The saved occupancy axes are X,Y,Z and are not transposed into a display frame before projection. `--reference-camera` can override the reference timestamp for diagnostics; it does not change the model. The original LiDAR branch's CAM_FRONT reference-time mismatch and moving-object time offsets remain limitations. These renderings visualize model predictions; they are not image segmentation ground truth or an accuracy evaluation.

Depth in the NPZ is camera Z in meters (0 for no hit); `voxel_index` is `(x * ny + y) * nz + z` (-1 for no hit). Rendering does not modify the saved occupancy. Projection adds runtime, so use `--no-camera-projection` on the inference command when only occupancy/BEV files are needed.

### Compare SAM3 segmentation with the occupancy projections

Use the saved EFFOcc frame and its original six camera images to produce a matched three-column comparison: **original / SAM3 overlay / EFFOcc occupancy overlay**. SAM3 runs independently on the full original image with the 16-class prompts in [nuscenes_classes.txt](../tasks/nuscenes_sam3/config/nuscenes_classes.txt). Shared semantic IDs 1..16 use the same palette and both overlays use the same opacity.

```bash
bash tasks/effocc/scripts/compare_effocc_sam3.sh \
  --input-dir outputs/effocc_mini_with_cameras \
  --max-samples 1 --alpha 0.45 \
  --output-dir outputs/effocc_sam3_demo
```

The source may be an EFFOcc run root or one sample directory. Completed `camera_projections/` are reused after checking sample and camera tokens. For projections saved elsewhere, pass `--projection-dir outputs/effocc_six_camera_projection`; if automatic projections are absent and no explicit projection directory is supplied, missing views are rendered before loading SAM3. EFFOcc inference is not rerun. Default `--max-samples 1` compares one saved LiDAR sample's six cameras; 0 processes all saved samples.

Optional parameters: `--sam3-checkpoint models/sam3/sam3.pt`, `--classes-file tasks/nuscenes_sam3/config/nuscenes_classes.txt` (retain the complete nuScenes16 taxonomy), `--score-threshold 0.5`, `--mask-threshold 0.5`, `--precision bf16`, `--device cuda`, and `--dataset-dir`. CPU SAM3 requires `--device cpu --precision fp32`. All run in the existing WSL `vla-supervised` environment.

```text
<comparison-output>/
  run_config.json
  <sample-name>/
    classes.json / comparison_config.json
    CAM_FRONT_original.jpg
    CAM_FRONT_sam3_class_ids.png       # uint8 0..16; 0 is unassigned
    CAM_FRONT_sam3_semantics.png       # White unassigned background
    CAM_FRONT_sam3_overlay.jpg         # Original-size independent SAM3 segmentation
    CAM_FRONT_sam3_instances.json
    CAM_FRONT_occupancy_class_ids.png  # Unchanged projected IDs 0..16/255
    CAM_FRONT_occupancy_semantics.png
    CAM_FRONT_occupancy_overlay.jpg
    CAM_FRONT_comparison.jpg           # Three columns for this exact camera frame
    six_cameras_sam3.jpg               # 3x2 SAM3 overview
    six_cameras_comparison.jpg         # Six rows x three columns, one camera per row
```

The same files are saved for every camera channel. Per-camera masks and overlays retain original dimensions; comparison panels are resized for viewing. **SAM3 0 means unassigned; EFFOcc 0 means occupied others; class17/free is transparent and projection255 means no ray hit.** Those labels are not merged. Input images are matched by sample token and camera token, so nearby frames are not substituted. `comparison_config.json` records both coverage fractions, thresholds in the run configuration, prompts, image paths and source projections. This is a visual comparison of two predictions, not GT evaluation or a claim that pixel differences measure either model's accuracy. EFFOcc sees its configured image crop and bounded occupancy volume, whereas SAM3 sees the full image.
