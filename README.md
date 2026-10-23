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

Classes are now read directly from the [custom prompt configuration](tasks/nuscenes_sam3/config/six_classes.txt), supporting additions such as crosswalk. The existing `run_six_classes.sh` entry name remains available. Sidewalk/road are planes by default; other classes retain observed points. Use `--plane-classes` to select surfaces explicitly:

```bash
bash tasks/nuscenes_sam3/scripts/run_six_classes.sh
```

**Final result (six-class mode):** sparse semantic BEV on the left, BEV with road/sidewalk planes in the middle, and observed/inferred/non-ground source evidence on the right. Standard16 also produces independent GT BEV and comparison images.

![nuScenes final result: semantic BEV, road and sidewalk planes, and source map](docs/images/nuscenes.png)

The default grid is **100m × 100m**, with **0.2m** cells and **500 × 500** pixels; native LiDAR +x points right and +y points up. The standard entry writes `outputs/nuscenes_sam3/<timestamp>/`, with a BEV section first in `result.html` and final images in `results/BEV/<frame>/`. Six-class output is under `outputs/nuscenes_six_classes/<timestamp>/`, with BEV first in its result page.

Prompts: [16-class configuration](tasks/nuscenes_sam3/config/nuscenes_classes.txt), [six-class configuration](tasks/nuscenes_sam3/config/six_classes.txt). Vehicle includes cars/trucks/buses/trailers; bike includes bicycles/motorcycles; rider means the riding person. The six-class mode does not reuse16-class metrics, and lidarseg has no independent rider GT.

### Graphical generator

Start the local generator when you prefer UI configuration:

```bash
bash tasks/nuscenes_sam3/scripts/run_generator.sh
```

On Windows, run or double-click `tasks\nuscenes_sam3\scripts\run_generator.bat`. Open **http://127.0.0.1:8765**. The service uses the existing WSL `vla-supervised` environment with no additional web framework.

Add/delete categories, edit English prompts (`|` separates alternatives), and mark each class as a plane. Select one or multiple mini scenes using checkboxes, descriptions, frame counts, train/val split, availability, search and filtering. The default is1 frame total across selected scenes; enable the all-frames checkbox to process all selected frames. Advanced settings include starting index, map resolution, planeZ, polygon edge limit, camera projection range andGPU/CPU.

Preview estimates or start directly. The page shows state, camera-image progress, live logs, stop control, history and result links. Only one generation job runs at a time. Each `outputs/nuscenes_generator/<run-id>/` independently stores `classes.txt`, `ui_config.json`, `job.json`, `run.log`, and `output/`; **the source prompt TXT is not overwritten**. Class/plane choices remain attached to the run. Stopping preserves partial outputs; complete viewers require successful completion. Result links work while the UI server is running; output HTML is also available offline.

Change the port with `--port 8766`, or specify `--dataset-dir`, `--classes-file`, and `--output-root` when launching. Closing the launch terminal ends the service.

### Minimum-area bounding boxes per instance

New scene HTML includes an **Instance boxes** checkbox in Layers, hidden by default. Enable it to show each instance's **minimum-area rotated BEV rectangle** in3D and BEV Map. Boxes are independent of point/surface layer switches and follow frame selection, all-frame mode and class filters. Plane-class instances also retain their own boxes; a semantic class is not pooled into one box.

The rectangle encloses every retained XY support point for that camera-local instance, without ROI cropping or artificial padding. It is horizontal at the median support Z. With SOR/DBSCAN it uses retained points; with filtering off it uses original instance support. Singleton/collinear support stays a point/segment with zero area; empty instances are saved without drawing. These are visible-support BEV footprints, not3D detector boxes or guaranteed full-object extents. Objects may repeat across cameras, and box IDs are not tracks.

No new generation flag is needed: the UI, `run_scenes.sh`, and custom segmentation entry export boxes automatically. Re-export saved results that contain instance point memberships:

```bash
bash tasks/nuscenes_sam3/scripts/run_scenes.sh \
  --input-dir outputs/instance_filter_combined_verified/segmentation \
  --point-filter sor_dbscan --dbscan-eps 1.2 \
  --output-dir outputs/scene_with_instance_boxes
```

Each `scenes/<scene>/<sample_token>_boxes.json` stores camera/instance identity, class, support point indices, corners, center, length/width, yaw and area. Scene `manifest.json` references the files and counts; HTML embeds matching geometry. Custom segmentation and instance-hull output additionally save `instance_boxes.json` in native LiDAR coordinates. Older results without actual instance membership do not substitute class-wide boxes: the control is disabled until segmentation is regenerated with memberships.

### Instance-support cleaning: SOR / DBSCAN

The generator UI adds an instance-point filter: off(default), SOR, DBSCAN, or SOR→DBSCAN, with expandable parameters. Each camera-local SAM3 instance's original3D support is filtered separately; same-class people/vehicles are not pooled into one cluster. Retained same-class camera memberships are unioned. Image masks and zero-plane IPM retain their predictions; colored point support, point polygons and their BEV Map use the filtered support. Removed points get display label0 while original coordinates/labels/indices remain stored.

```bash
bash tasks/nuscenes_sam3/scripts/run_scenes.sh \
  --scenes scene-0103 --samples 1 \
  --plane-classes sidewalk road crosswalk \
  --point-filter sor_dbscan \
  --sor-k 16 --sor-std 2 \
  --dbscan-eps 1.2 --dbscan-min-samples 5 \
  --cluster-policy significant --min-cluster-size 10 --cluster-ratio 0.1 \
  --output-dir outputs/filtered_scene
```

`--point-filter` accepts `none`, `sor`, `dbscan`, `sor_dbscan`; the same flags work with `run_six_classes.sh` and `run_instances.sh`. SOR removes points whose mean k-neighbor distance exceeds population mean + sor_std × population standard deviation. Self is excluded, k is capped by point count, and SOR retains sets of at most2 points. DBSCAN eps is in meters and min-samples includes self. `largest` retains the biggest cluster; `significant` retains clusters at least `max(min-cluster-size, largest-cluster-size×cluster-ratio)`. Noise and rejected components are removed. No valid cluster can mean an empty instance; no silent fallback restores its original points.

**Real sparse distant targets can also be removed**, especially crossings/pedestrians with little support. Start with SOR or looser density/cluster thresholds and inspect before/after evidence; filtering is not an accuracy guarantee. Scene indexes link cleaning comparisons. `scenes/<scene>/cleaning/<sample>/` stores original/filtered labels, keep masks, removed point indices, per-instance JSON and red/green comparison. New segmentation saves `point_instances.json` so `run_scenes.sh --input-dir ... --point-filter ...` can change algorithms without another SAM3 run. Older outputs without actual instance membership require regeneration rather than treating semantic class groups as instances.

### Choose plane classes, including crosswalk

The prompt TXT specifies **which categories to recognize**; `--plane-classes` selects **which of them become surfaces**. The added crosswalk prompt is retained. Current seven-class IDs follow file order: sidewalk1, road2, crosswalk3, pedestrian4, vehicle5, bike6, rider7. Colors and rider/person overlap handling follow names rather than old fixed IDs.

```bash
# New segmentation: road, sidewalk and crosswalk all become surfaces and BEV Map labels.
bash tasks/nuscenes_sam3/scripts/run_scenes.sh \
  --scenes scene-0103 --samples 0 \
  --plane-classes sidewalk road crosswalk \
  --output-dir outputs/scene_with_crosswalk_planes

# Reuse segmentation containing crosswalk: only road becomes a plane; other classes remain colored points.
bash tasks/nuscenes_sam3/scripts/run_scenes.sh \
  --input-dir outputs/crosswalk_planes_verified/segmentation \
  --plane-classes road \
  --output-dir outputs/road_plane_only
```

The selection controls zero-plane IPM, point polygons, BEV Map and HTML; selected plane classes keep separate labels. It also works with `run_six_classes.sh --plane-classes ...` and `tasks/flat_ground/scripts/run.sh --plane-classes ...` (legacy `--project-classes` is an alias). Crosswalk becomes a plane only when selected. New runs default to sidewalk/road; saved-input exports reuse their saved plane selection unless overridden. Nonselected recognized classes remain in segmentation/points and can be hidden through HTML class controls.

Every output stores its own `classes.json`, taxonomy and plane_class_ids. Saved IDs are interpreted using that schema: editing the TXT cannot turn old pedestrian3 into new crosswalk3. Adding crosswalk to a run that never segmented it requires fresh segmentation. Outputs remain single-label masks; accepted crosswalk masks override road/unassigned pixels so the larger road mask does not erase the crossing, subject to prompt-segmentation errors.

### Scene HTML: two road surfaces, colored points, and a timeline

The six-class entry now automatically creates `scenes/index.html` and a standalone offline `scenes/<scene>/viewer.html` per scene. Toggle or overlay the **zero-plane IPM** and **BEV point-polygon** road/sidewalk surfaces. Pedestrian, vehicle, bike, and rider retain their original colored3D points rather than being expanded into object planes.

- Timeline scrubbing, previous/next, play/pause and FPS; alternatively overlay every frame embedded in that HTML.
- Independent controls for each surface method, semantic points, ground support points and ego trajectory; separate sidewalk/road/pedestrian/vehicle/bike/rider visibility, with unassigned points hidden by default.
- Mouse orbit, right/Shift-drag pan, wheel zoom; top/3D/fit buttons, point size and surface opacity.

Generate a complete selected scene (`--samples 0` means all frames in the current selection):

```bash
bash tasks/nuscenes_sam3/scripts/run_scenes.sh \
  --scenes scene-0103 --samples 0 \
  --output-dir outputs/my_scene_viewer
```

Reuse six-class results without new segmentation:

```bash
bash tasks/nuscenes_sam3/scripts/run_scenes.sh \
  --input-dir outputs/nuscenes_six_classes/20261016_135902_151161 \
  --output-dir outputs/my_saved_scene_viewer
```

With `--input-dir`, all saved frames are exported by default; without it, the default is one frame. Multiple scenes produce separate HTML files. The index reports exported frames versus dataset scene frames, distinguishing partial from complete scenes. The six-class `result.html` links to its scene viewers; `--no-scene-viewer` skips this additional export.

Both surfaces and point clouds use each sensor timestamp's pose and are aligned to the first displayed frame's ego XY/yaw scene frame, in meters. IPM stays at global `z=0`. Point polygons retain the0.2m grid and bounded6m-edge interpolation, then are placed as horizontal patches at each class's median observed global height. All original3D points and indices are retained, without sampling or flattening. All-frame overlay keeps moving-object trails and performs no tracking. Per-frame NPZ and scene `manifest.json` save transforms, labels, exact row-rectangle/triangle meshes and support counts. Meshing preserves grid holes instead of bridging them with a convex hull. HTML embeds the full data with no CDN/server dependency; file and GPU memory use grow with frame count.

### BEV Map: divide ground surfaces into metric cells

The scene viewer adds a **BEV Map** button to switch between the original3D scene and an orthographic semantic grid. IPM and point-polygon surfaces are independently sampled into a common scene XY grid, default **0.2m/cell**. Single-frame mode shows the current frame; all-frame mode shows the fused scene map. The map-method selector chooses IPM/Polygon independently of3D layers. Class switches control road/sidewalk cells and other semantic point overlays. Drag to pan, wheel to zoom, show the fine grid, or exportPNG.

New `run_scenes.sh` and six-class runs include maps automatically. Upgrade saved scenes without segmentation or reprojection:

```bash
bash tasks/nuscenes_sam3/scripts/run_bev_map.sh \
  --input-dir outputs/nuscenes_scene0103_complete/scenes \
  --map-resolution 0.2 \
  --output-dir outputs/nuscenes_scene0103_bev_map
```

Use `run_scenes.sh --map-resolution 0.4` to change a new export's map cell size. Scene maps span the complete surface extent instead of a fixed single-frame500x500: the40-frame scene-0103 example is **1086x597 cells,217.2mx119.4m**. The frame remains the first ego XY/yaw, +X right and +Y up.

Each target cell center inverse-samples the original plane grid without extra dilation/interpolation/hole filling. Each frame casts at most one vote per cell; the most frequent class wins, exact ties choose sidewalk(ID1), and class0 abstains. `conflict` marks cells receiving both road and sidewalk evidence. `observation_count` counts frames with a surface label, **not measured ground observations**. Unsupported space is not free space. Map display collapses height; original3D point/surface data remain available.

Each scene's `bev_map/` contains `ipm_map.png` / `polygon_map.png`, `*_fused.npz` (labels, class_votes, observation_count, conflict, tie), `*_conflicts.png`, per-frame `<sample_token>.npz`, and `metadata.json`. HTML embeds current/fused labels for offline use. Upgraded exports retain the original scene numeric-data path for future re-exports.

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
