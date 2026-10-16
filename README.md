# vla-supervised-ad

**English** | [简体中文](README.zh-CN.md)

Use local SAM3 and EFFOcc for image segmentation, semantic point-cloud BEV, occupancy prediction, and flat-ground projection. The project has **five task categories**, each with a one-command entry point and a final image from an actual run below.

Run commands from the project root in **WSL Bash**. Launchers select the existing Conda **`vla-supervised`** environment and local weights. Each task defaults to one image or one complete six-camera frame.

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad
```

| Task | Input and objective | One-command entry point |
|---|---|---|
| [1. Sidewalk Semantic](#task-sidewalk) | Image segmentation compared with pixel GT | `bash tasks/sidewalk/scripts/run.sh` |
| [2. nuImages](#task-nuimages) | Annotated camera-image segmentation and evaluation | `bash tasks/nuimages/scripts/run.sh` |
| [3. nuScenes SAM3 and BEV](#task-nuscenes) | Six cameras, LiDAR semantic fusion, road/sidewalk surfaces | `bash tasks/nuscenes_sam3/scripts/run.sh` |
| [4. EFFOcc versus SAM3](#task-effocc) | Image + LiDAR occupancy, projected into the same camera frames | `bash tasks/effocc/scripts/run.sh` |
| [5. Flat-ground semantic IPM](#task-flat-ground) | Direct camera-mask projection onto assumed global z=0 ground | `bash tasks/flat_ground/scripts/run.sh` |

Installation, all parameters, dataset paths, and evaluation conventions are retained in the [English detailed guide](docs/guide.en.md) and [中文详细指南](docs/guide.zh-CN.md). Images below are unmodified copies of actual outputs in [docs/images](docs/images/), independent of ignored experiment directories.

<a id="task-sidewalk"></a>
## 1. Sidewalk Semantic: segmentation and pixel evaluation

SAM3 segments images using the Sidewalk dataset taxonomy, producing class-ID maps, overlays, dataset-GT comparisons, and pixel metrics including mIoU and per-class IoU. The default processes one image.

```bash
bash tasks/sidewalk/scripts/run.sh
```

**Final result:** original image, SAM3 prediction, and dataset ground truth for the same sample.

![Sidewalk Semantic final result: original, SAM3 segmentation and GT](docs/images/sidewalk.jpg)

Prompts: [sidewalk_classes.txt](tasks/sidewalk/config/sidewalk_classes.txt). Outputs are under `outputs/sidewalk/<timestamp>/`: open `result.html`; final images are in `results/`. Custom classes and evaluation rules are documented in the [detailed guide](docs/guide.en.md#1-sidewalk-semantic-classes-and-evaluation).

<a id="task-nuimages"></a>
## 2. nuImages: camera segmentation and GT comparison

SAM3 segments annotated nuImages camera keyframes, decodes dataset masks, and produces predictions, GT comparisons, and error maps with evaluation on valid annotated pixels. The default processes one image; a sample does not need annotations for all six views.

```bash
bash tasks/nuimages/scripts/run.sh
```

**Final result:** top row: original, SAM3 overlay, and prediction labels; bottom row: GT overlay, GT labels, and prediction errors.

![nuImages final result: SAM3, dataset ground truth and errors](docs/images/nuimages.jpg)

Prompts: [nuimages_classes.txt](tasks/nuimages/config/nuimages_classes.txt). Outputs are under `outputs/nuimages/<timestamp>/`, with `result.html` and `results/`. Original classes, ego handling, and ignore regions are described in the [detailed guide](docs/guide.en.md#5-nuimages-image-segmentation-ground-truth-comparison-evaluation-and-visualization).

<a id="task-nuscenes"></a>
## 3. nuScenes SAM3: six cameras, semantic BEV, and ground surfaces

After six-camera SAM3 segmentation, calibration and poses associate image labels with original LiDAR points. Camera predictions are fused into semantic BEV. Road and sidewalk can be interpolated into continuous patches using locally supported bounded triangles, with interpolation sources saved separately.

```bash
# Standard16: six cameras, sparse lidarseg evaluation, semantic BEV and ground planes.
bash tasks/nuscenes_sam3/scripts/run.sh
```

For only **sidewalk, road, pedestrian, vehicle, bike, and rider**, use the six-class mode within this same task category. Only sidewalk/road become surfaces; the other classes retain observed semantic evidence:

```bash
bash tasks/nuscenes_sam3/scripts/run_six_classes.sh
```

**Final result (six-class mode):** sparse semantic BEV on the left, BEV with road/sidewalk planes in the middle, and observed/inferred/non-ground source evidence on the right. Standard16 also produces independent GT BEV and comparison images.

![nuScenes final result: semantic BEV, road and sidewalk planes, and source map](docs/images/nuscenes.png)

The default grid is **100m × 100m**, with **0.2m** cells and **500 × 500** pixels; native LiDAR +x points right and +y points up. The standard entry writes `outputs/nuscenes_sam3/<timestamp>/`, with a BEV section first in `result.html` and final images in `results/BEV/<frame>/`. Six-class output is under `outputs/nuscenes_six_classes/<timestamp>/`, with BEV first in its result page.

Prompts: [16-class configuration](tasks/nuscenes_sam3/config/nuscenes_classes.txt), [six-class configuration](tasks/nuscenes_sam3/config/six_classes.txt). Vehicle includes cars/trucks/buses/trailers; bike includes bicycles/motorcycles; rider means the riding person. The six-class mode does not reuse16-class metrics, and lidarseg has no independent rider GT.

Extensions belong to this same task:

| Extension | Entry point | Purpose |
|---|---|---|
| Instance-point convex hulls | `bash tasks/nuscenes_sam3/scripts/run_instances.sh` | Preserve camera instances, cover assigned points with hulls, and save separate association/ground results |
| Merge saved cross-camera instances | `bash tasks/nuscenes_sam3/scripts/merge_instances.sh --input-dir <saved-instance-output>` | Match same-class foreground by shared points, deduplicate unions, and recompute hulls |
| Ground planes from saved results | `bash tasks/nuscenes_sam3/scripts/run_ground.sh --input-dir <saved-output>` | Reuse segmentation/point predictions without rerunning SAM3 |
| Refresh an older one-command result | `bash tasks/nuscenes_sam3/scripts/run.sh --refresh-results --output-dir <existing-one-command-output>` | Add ground products and collect BEV images |

Interpolated surfaces are not new observations, and sparse projected-point evaluation is not dense image GT. Fusion, boundaries, and evaluation are explained in the [detailed guide](docs/guide.en.md).

<a id="task-effocc"></a>
## 4. EFFOcc: occupancy prediction and same-frame SAM3 comparison

EFFOcc Fusion-R18 uses six cameras plus LiDAR/history sweeps to predict3D semantic occupancy. Occupancy is raycast back into all six cameras and compared with independent SAM3 segmentation. The one-command entry runs both model stages and builds comparisons.

```bash
bash tasks/effocc/scripts/run.sh
```

**Final result (front-camera example):** original image on the left, SAM3 segmentation in the middle, and same-frame EFFOcc occupancy projection on the right. Complete six-camera comparisons are also generated.

![EFFOcc final result: original, SAM3 and same-frame occupancy projection](docs/images/effocc.jpg)

Outputs are under `outputs/effocc/<timestamp>/`: `result.html` shows final comparisons, `results/` collects images, `raw/` retains occupancy/projections, and `comparison/` retains SAM3 comparisons. SAM3 uses the [nuScenes16 prompt file](tasks/nuscenes_sam3/config/nuscenes_classes.txt). EFFOcc itself has no text prompts; its taxonomy is fixed by trained weights.

The current inference-only PyTorch adaptation strictly loads the checkpoint but has no full numerical-equivalence validation against the original MMCV/spconv backend. Dense occupancy GT is unavailable locally; prediction differences are visual comparisons, not accuracy scores. See the [EFFOcc guide](docs/guide.en.md#6-effocc-fusion-r18-on-nuscenes-mini).

<a id="task-flat-ground"></a>
## 5. Flat-ground semantic IPM: direct projection onto z=0

Assume a horizontal plane at **nuScenes global-map z=0**. Camera calibration and each image's pose map SAM3 semantic regions directly onto this plane. IPM inverse-samples camera pixels at BEV cell centers; it reads no LiDAR points and performs no triangle interpolation. Road and sidewalk are projected by default.

```bash
bash tasks/flat_ground/scripts/run.sh
```

**Final result:** fused semantic ground BEV on the left, geometric camera coverage in the middle, and the camera supplying each selected label on the right. This frame differs from task3's native LiDAR frame: +X is ego forward (right in the image), and +Y is left (up in the image).

![Flat-ground final result: semantic plane, camera coverage and source camera](docs/images/flat_ground.png)

The default remains **100m × 100m,0.2m cells,500 × 500**. Results are under `outputs/flat_ground/<timestamp>/`; open `result.html`. Reuse saved six-class masks from task3 without rerunning inference:

```bash
bash tasks/flat_ground/scripts/run.sh \
  --input-dir outputs/nuscenes_six_class_planes \
  --plane-z 0 \
  --output-dir outputs/flat_ground_demo
```

`--plane-z` is global-map height, not absolute sea-level altitude. Slopes, curb heights, occlusion, and segmentation errors distort the projection; white cells lack an accepted ground label. With `--project-classes sidewalk road pedestrian vehicle bike rider`, elevated objects are flattened and stretched, not converted into true contact footprints. Formulas, options, and evidence files are described in the [flat-ground IPM guide](docs/guide.en.md#flat-ground-ipm-project-semantic-regions-directly-onto-z0).

## Environment, parameters, and details

- Use WSL's `vla-supervised` environment; [environment setup](docs/guide.en.md#environment-setup) retains installation, dependency, and dataset checks.
- Each task defaults to one sample. Use `--samples N` to change count, `--start-index I` for the starting position, and a new/empty `--output-dir`; each entry's `--help` lists its actual parameters.
- Sidewalk, nuImages, standard nuScenes, and EFFOcc collect final images in `results/`. Six-class and flat-ground tasks save final images per frame and also provide `result.html`.
- Run project verification with `bash shared/scripts/test_all.sh`. Environment, weights, datasets, and historical experiment outputs are not moved by the documentation changes.

[Full command reference](docs/guide.en.md#command-reference) · [Project layout and dataset paths](docs/guide.en.md#project-layout-and-data-paths) · [中文详细指南](docs/guide.zh-CN.md)
