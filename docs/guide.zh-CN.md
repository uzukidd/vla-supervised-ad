# vla-supervised-ad

[English](guide.en.md) | **简体中文**

使用本地 SAM3 对 Sidewalk Semantic、nuScenes mini 和 nuImages 相机图像做文本提示分割，并导出类别图、可视化和评估指标。nuScenes 支持 lidarseg 投影评估、原始点融合与二维 BEV 导出，nuImages 使用图像掩码真值评估。

阅读顺序：[环境配置](#环境配置) → [可用命令总表](#可用命令总表) → [项目目录与数据路径](#项目目录与数据路径)。后续章节详细说明各数据集、点云输出和评估口径。

## 实例最小矩形框

场景查看器新增默认隐藏的“实例矩形框”图层，3D/BEV Map均支持，随帧/全帧/类别选择，独立于点云和平面开关。每个相机局部instance使用当前清洗后支持点的最小面积旋转矩形，空/点/线退化记录保留；保存角点、尺寸、方向、面积和原始点索引，不当作检测真值或轨迹。新管线自动导出，已有实例结果无需重跑模型即可重建。见 [README实例矩形框](../README.zh-CN.md#每个-instance-的最小矩形-bounding-box)。

## 实例点SOR/DBSCAN清洗

生成器UI可选择none/sor/dbscan/sor_dbscan；CLI使用 `--point-filter`，配合SOR邻域/标准差、DBSCAN半径/核心点数和团块保留策略。按真实相机局部instance处理，原始数组保留，点多边形与点云语义使用保留支持点，IPM不变。旧结果缺少实例点归属需要重新分割。参数、团块保留规则及红绿对比输出见 [README实例清洗](../README.zh-CN.md#instance-支持点清洗sor--dbscan)。下方历史“无离群点处理”描述以当前可选配置为准，默认关闭。

## 本地生成器UI

运行 `bash tasks/nuscenes_sam3/scripts/run_generator.sh`（Windows为同目录run_generator.bat），打开 http://127.0.0.1:8765。页面可编辑类别/提示词、勾选平面与mini场景，预览或启动生成、查看日志/进度、停止运行及打开结果。每次保存独立配置，不修改默认TXT。详细用法见 [README 图形化生成器](../README.zh-CN.md#图形化生成器)。

## 可配置类别与平面选择

最新自定义管线不再固定六类。提示词TXT新增crosswalk或其它名称后自动按行分配ID；用 `--plane-classes sidewalk road crosswalk` 指定其中哪些生成平面，未列入类别保留染色点云。参数贯通run_scenes.sh、run_six_classes.sh和flat_ground，BEV Map与HTML按输出保存的动态类别表和plane_class_ids生成。旧文件名仅作兼容入口，下方“六类/仅road、sidewalk”的历史说明以保存结果的类别表及当前参数为准。完整命令与旧ID保护见 [README 自选平面类别](../README.zh-CN.md#自选平面类别含-crosswalk)。

## BEV Map 栅格地图模式

场景HTML的 **BEV Map** 按钮将两种地面平面分别采样为同一个米制栅格，支持当前帧/全帧投票融合、方法选择、类别隐藏和原始点云XY叠加；未分配格留白。默认0.2米/格，保存标签、每类帧票数、覆盖帧数和冲突；不会把缺证据格变成自由空间。新管线自动包含，已有场景使用 `bash tasks/nuscenes_sam3/scripts/run_bev_map.sh --input-dir <已有场景目录> --output-dir <新目录>` 升级。完整参数/文件说明见 [README BEV Map](../README.zh-CN.md#bev-map将路面分成固定米制格子)。

## 交互式场景时间轴

六类管线现在自动生成每场景一个离线3D HTML，同时提供零水平面IPM、BEV点多边形道路/人行道、原始分类点云、播放/全部帧和图层按钮。完整用法与坐标定义见 [README 场景说明](../README.zh-CN.md#场景-html两种路面染色点云与时间轴)。已有六类输出可执行 `bash tasks/nuscenes_sam3/scripts/run_scenes.sh --input-dir <已有六类输出> --output-dir <新目录>`，不重跑分割。场景HTML是全量数据副本，不依赖CDN；索引标明部分/完整场景。

## 每个任务只运行一条命令

日常使用只需从下面选择 **一条命令**。入口会自动使用现有WSL `vla-supervised` 环境，完成分割、适用的投影/评估和对比图，不需要手动串联高级脚本。

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad
```

| 任务 | 运行命令 | 最终对比 |
|---|---|---|
| EFFOcc + SAM3 | `bash tasks/effocc/scripts/run.sh` | 同帧六相机：原图 / SAM3 / EFFOcc占用投影叠加 |
| 平地相机直接投影（新任务） | `bash tasks/flat_ground/scripts/run.sh` | SAM3道路/人行道区域直接逆投影到全局z=0，不用LiDAR |
| 六类分割 + 道路/人行道铺面 | `bash tasks/nuscenes_sam3/scripts/run_six_classes.sh` | sidewalk / road / pedestrian / vehicle / bike / rider，六相机加500x500 BEV |
| nuScenes SAM3 | `bash tasks/nuscenes_sam3/scripts/run.sh` | 六相机SAM3 / 稀疏lidarseg GT，加500x500语义BEV对比 |
| 连续道路/人行道平面（已有单帧） | `bash tasks/nuscenes_sam3/scripts/run_ground.sh` | 合并六相机地面区域，输出稀疏/平面/填充来源对比 |
| SAM3实例多边形（nuScenes mini实验） | `bash tasks/nuscenes_sam3/scripts/run_instances.sh` | 单帧六相机：实例掩码 / 对应LiDAR点 / 最小凸BEV多边形 |
| nuImages | `bash tasks/nuimages/scripts/run.sh` | 原图 / SAM3 / 数据集GT / 错误图 |
| Sidewalk | `bash tasks/sidewalk/scripts/run.sh` | 原图 / SAM3 / 数据集GT |

**默认只处理一个样本：Sidewalk/nuImages为一张图，nuScenes/EFFOcc为一帧完整六相机。** 每次自动创建 `outputs/<任务名>/<时间戳>/`，重复运行不会覆盖旧结果。直接用浏览器打开末尾打印的 **`result.html`**；最终对比图片集中放在 **`results/`**。数值掩码、中间预测和详细元数据保留在 `raw/`，EFFOcc的SAM3对比同时保留在 `comparison/`。

**nuScenes的BEV也在同一个一键流程中。** `bash tasks/nuscenes_sam3/scripts/run.sh` 现在同时生成原始语义/GT BEV和连续道路、人行道平面。打开 `result.html` 顶部的 **BEV俯视图** 分区即可；每帧六张BEV/地面PNG还集中保存到 `results/BEV/<frame>/`：`bev_prediction.png`、`bev_ground_truth.png`、`bev_comparison.png`、`ground_plane.png`、`ground_sources.png`、`ground_comparison.png`。六相机对比放在独立分区。缺少必要BEV产物时汇总会报错，不会只输出相机图片却标记成功。

旧的一键运行结果无需重跑SAM3即可补齐：

```bash
bash tasks/nuscenes_sam3/scripts/run.sh \
  --refresh-results \
  --output-dir outputs/nuscenes_sam3/20261016_102509_921512
```

将路径替换成自己已有的输出目录。刷新从保存的点预测补地面图，保留原始预测/指标并重建结果索引；新运行仍只需普通 `run.sh`。普通语义管线使用已融合的点类别补面，独立实例实验则先按掩码分数确定类别；两者均记录来源，不使用GT填充。

需要时只改以下参数：

```bash
# 处理三帧六相机，显式指定一个新输出目录。
bash tasks/effocc/scripts/run.sh --samples 3 --output-dir outputs/my_effocc_comparison

# 从第1帧开始，该帧已有真实历史LiDAR扫帧。
bash tasks/effocc/scripts/run.sh --start-index 1
```

`--samples` 对nuScenes/EFFOcc始终表示完整六相机帧数，入口自动转换图像数量；`--start-index` 使用同样的任务单位。默认CUDA；`--device cpu` 自动选用兼容的float32 SAM3模式。EFFOcc各阶段使用独立进程，先释放EFFOcc模型显存，再加载SAM3。

EFFOcc对比两种独立模型预测，其它任务将SAM3与数据集已有GT比较。这个入口不会给Sidewalk/nuImages凭空增加第二个模型，不把稀疏lidarseg当作密集图像真值，也不会把EFFOcc和SAM3差异当成准确率。已有的数据集指标与最终图片一起汇总。

下方保留环境说明和 **高级分步入口**，需要改类别、数据路径等设置时再查阅，原来的命令仍可使用。

## 平地IPM：把语义块直接投到 z=0

这是 [tasks/flat_ground](../tasks/flat_ground/) 中的独立任务，与LiDAR点云栅格化、三角网/凸包补面分开。假设地面为 **nuScenes全局地图坐标 z=0米**，不是大地测量意义的海平面海拔。显示坐标以当前CAM_FRONT时刻的自车XY位置为中心，仅保留yaw：局部+X为前方、+Y为左侧；PNG列沿+X、行沿-Y。默认100米x100米、0.2米格、500x500。

```bash
# 对一帧六相机运行SAM3，再直接投影道路/人行道掩码。
bash tasks/flat_ground/scripts/run.sh

# 复用已有六类掩码，不运行SAM3，也不读取点云。
bash tasks/flat_ground/scripts/run.sh \
  --input-dir outputs/nuscenes_six_class_planes \
  --plane-z 0 \
  --output-dir outputs/flat_ground_demo
```

默认结果是 `outputs/flat_ground/<时间戳>/result.html`，包括合并BEV/覆盖/来源相机对比，以及六个相机的原图/SAM3/单相机IPM。可用 `--samples 1`、`--start-index`、`--split mini_val/mini_train/all`、`--scenes`、`--plane-z`、`--bev-roi-size`、`--bev-resolution`、`--max-distance`（默认相机到地面斜距80米）、`--output-dir`。有 `--input-dir` 时索引选择已有掩码帧，核对精确sample/camera token；没有时用本地SAM3及六类配置直接推理。独立读取相机元数据，不需要LiDAR samples、sweeps或lidarseg表。

像素射线和平面的交点为 `p_global = camera_origin + t * ray_global`，其中 `t = (plane_z - camera_origin_z) / ray_global_z`。平行、反向/相机后方、过近或超距交点排除。为避免像素正向撒点产生孔洞，实现逐个BEV **格子中心** 的全局z=plane_z点投到各原相机，再取最近语义像素。这是同一平面单应几何的逆采样，不使用点云深度或学习深度；非地面标签不会变成道路，未分配掩码或无有效相机对应仍留白。

默认只投 `sidewalk`、`road`。若确实要强行把所有类别拍到地面，可加 `--project-classes sidewalk road pedestrian vehicle bike rider`，但高于地面的行人/车辆/两轮车会变成拉伸的轮廓，不是真实接地足迹。多相机重叠处选择地面射线入射余弦最大的有效类别证据（更正对/更近的平面视角），平票按相机顺序；`camera_count` 单独记录几何覆盖，不与类别证据混淆。不做凸包、三角网插值或额外填洞。

```text
<flat-ground-output>/
  result.html / run_config.json
  <scene_frame_sample>/
    bev_ground_ipm.png            # 500x500假设地面语义图
    bev_class_ids.png             # ID0..6，0为未分配/无效
    bev_camera_coverage.png       # 几何相机视角数量1..6
    bev_source_camera.png         # 最终语义证据的来源相机
    bev_comparison.png
    ipm_grid.npz                  # labels、source_camera、camera_count、support_count、incidence、XY网格及参考变换
    ipm_metadata.json / classes.json
    cameras/<camera>_ipm.png
    cameras/<camera>_class_ids.png / <camera>_source.json
    cameras/<camera>_sampled_bev_ids.png
    cameras/<camera>_geometry.npz # 每格pixel_xy/uv/depth/slant_distance/incidence/valid、内外参
    cameras/<camera>_comparison.jpg
```

**这是平地假设投影，不是重建表面或真值。** 坡度、凸起人行道、遮挡和位姿/掩码误差都会造成变形；全局z=0不保证就是真实路面。原图使用各自时间戳及位姿，未观测/无效格保持空白；几何可见不代表无遮挡，也不输出投影精度或可通行分数。原有LiDAR支撑的任务保持不变。

## 六类语义分割与 road、sidewalk 铺面

只需要六类时使用这个独立入口，原nuScenes16和EFFOcc流程保留：

```bash
bash tasks/nuscenes_sam3/scripts/run_six_classes.sh
```

默认一帧完整六相机mini_val、100米x100米ROI、0.2米格子（500x500）。末尾打印的 `outputs/nuscenes_six_classes/<时间戳>/result.html` 先展示语义BEV，再展示各相机原图/叠加/类别图。`--samples` 改帧数，`--start-index` 改起始帧，`--output-dir` 可指定新输出目录。

| ID | 输出类别 | 定义/默认提示词范围 |
|---|---|---|
| 0 | unassigned | 没有接受的mask，彩色图留白 |
| 1 | sidewalk | 人行道表面 |
| 2 | road | 道路表面，不是车道线实例 |
| 3 | pedestrian | 步行的人，正确拼写为pedestrian |
| 4 | vehicle | 汽车、卡车、公交、挂车，不包含bike |
| 5 | bike | 自行车或摩托车，骑乘者本人另分类 |
| 6 | rider | 骑自行车或摩托车的人 |

修改 [six_classes.txt](../tasks/nuscenes_sam3/config/six_classes.txt)，左侧类别名和顺序保留，右侧英文提示词可改，`|` 分隔别名。只提示这六类概念。一般mask重叠按实例置信度；已接受的rider掩码仅优先覆盖pedestrian或未分配像素，不覆盖bike/vehicle。文本提示仍可能混淆骑乘者、行人和车辆边界，类别掩码及实例元数据保留决定供核查。

六视角投影到原始LiDAR后按唯一点的非零类别多数票融合。只有 **sidewalk1和road2** 使用已有的有限边长地面三角网铺面（`--max-edge 6`，单位米）；其它类仅保留观测语义格，并在其支持格阻止铺面。最终图保持pedestrian/vehicle/bike/rider语义格，只填由道路/人行道三角形支持的地面、未分配或无点格，不扩大物体、不填满所有未知空间。`plane_sources.png` 的蓝色表示推断填充，不是实测表面或安全通行证据。

每帧包含 `six_class_comparison.png`（稀疏BEV/铺面BEV/来源图）、`bev_six_sparse.png`、`bev_six_plane.png`、`road_sidewalk_plane.png`、`plane_sources.png`、`six_class_plane.npz`、`plane_metadata.json`、完整 `points.npz`，以及 `cameras/<channel>_class_ids.png/_semantics.png/_overlay.jpg/_comparison.jpg/_instances.json/_source.json`。所有公开标签保持ID0..6（255为无点栅格）；仅三角网helper内部使用nuScenes16映射，不把内部ID当作输出类别。

不直接输出原nuScenes16或Occ3D指标：vehicle/bike分组已改变，lidarseg也没有独立rider标签，正式评估需要明确GT合并和骑乘者标注规则。未提示的视觉概念保持未分配；六类表示可用标签，不保证每帧六类都出现。`--help` 可看网格/铺面参数，默认SAM3阈值0.5，GPU精度bf16。

## 道路和人行道：跨相机连续地面区域

地面类别按语义区域融合，不按车辆/行人的实例身份处理。六相机同类点去重并统一到原始LiDAR XY，保留道路11、其它平面12、人行道13、自然地面14四类；共享点类别冲突按掩码置信度选择，平票按较小原实例ID。不同地面类别不会被合成同一类别。

已有实验结果可以直接补成平面，无需重跑SAM3：

```bash
bash tasks/nuscenes_sam3/scripts/run_ground.sh \
  --input-dir outputs/nuscenes_instance_hulls_experiment \
  --samples 1 --max-edge 6 \
  --output-dir outputs/ground_plane_demo
```

新运行 `bash tasks/nuscenes_sam3/scripts/run_instances.sh` 也会默认在每帧 `ground/` 中生成这些图，并优先展示到 `result.html`。`--ground-max-edge 6` 调整连接距离，`--no-ground-plane` 可关闭；已有结果独立转换的相同参数叫 `--max-edge`。

在500x500、0.2米格子内合并地面证据，再对各类观测格中心建立Delaunay三角网，仅填充 **所有边长不超过6米** 的三角形。这是局部边界内补面，不用全类别大凸包横跨整片未知区域。现有其它地面类别格不被覆盖；具有非地面语义点的格子阻止地面填充。推断区域冲突取最近的同类支持格，精确平票取较小类别ID。

输出包括 `ground_observed.png`（跨相机稀疏地面）、`ground_plane.png`（连续语义平面）、`ground_sources.png`（绿=真实观测格，蓝=推断填充格，红=非地面证据阻挡）、`ground_comparison.png`、`ground_grid.npz`（observed/filled/inferred/blocked及计数）、`semantic_points.npz`（去重类别点）和 `ground_regions.json`（各类边界轮廓、孔洞、接受三角形和元数据）。图像仍是100米x100米、500x500。道路/人行道各自的连通区域融合成片，但距离很远或无证据的分离区域不会强行连成一个多边形。

**补出来的平面不是观测真值。** 单帧点稀疏，有限边长三角形仍可能跨越未观测障碍；没有做地面高度/坡度估计或交通规则验证。本实验单独标记补面来源，不宣称安全可通行，不用原稀疏标签伪造密集平面精度。

## SAM3实例点云凸包实验

SAM3会为文本提示产生逐实例掩码，已有语义流程把掩码合并；本实验保留原掩码，把同一关键帧的投影LiDAR点分配给各检测实例，再计算 **覆盖100米x100米XY ROI内全部分配点的最小凸多边形**。这里采用凸包，不是任意形状的最少顶点多边形，也不是拟合三维包围框。

```bash
bash tasks/nuscenes_sam3/scripts/run_instances.sh
```

入口默认只跑 **mini_val一帧完整六相机**，使用nuScenes16类别提示词、100米x100米ROI和0.2米显示粒度（500x500）。结果位于 `outputs/nuscenes_instance_hulls/<时间戳>/result.html`。可选 `--start-index`、`--samples`、`--classes-file`、`--score-threshold`、`--mask-threshold`、`--bev-roi-size`、`--bev-resolution`、`--output-dir`；TXT须保留nuScenes16类别名。指定新目录复现实验：

```bash
bash tasks/nuscenes_sam3/scripts/run_instances.sh \
  --samples 1 --bev-resolution 0.2 --bev-roi-size 100 \
  --output-dir outputs/instance_hulls_demo
```

共享SAM3推理在语义合并前输出原始实例掩码。同一相机里被多个掩码覆盖的像素/点分配给置信度最高实例，完全平票保留较早检测，合并前掩码以packed bits保存。点分配使用标定、各传感器ego pose/时间戳的LiDAR到相机投影，同像素碰撞保留最近深度点。**原始实例仍是相机局部检测，不是轨迹。** 原ID/掩码与共享点保留，实验另外输出受约束的共享点关联结果：只比较不同相机中同类别的前景实例，它仍是身份关联假设，不是已验证的跟踪。

主凸包图使用前景类别1..10（车辆、行人等前景）；类别11..16（道路、人行道、自然地面、人造结构和植被等）另放在全部区域图中。这些SAM3掩码是语义区域，不能据此认定物理实例身份。相机图与BEV按实例ID统一颜色，不按类别统一颜色。

```text
<instance-output>/
  result.html / run_config.json
  <scene_index_sample>/
    bev_instance_points.png       # 分配点按相机局部实例ID染色
    bev_instance_hulls.png        # 前景凸包、原始支持点和ID标签
    bev_all_regions.png           # 前景加语义stuff区域凸包
    bev_instance_comparison.png   # 原点/分配点/前景凸包/全部区域
    instance_hulls.json           # 相机/实例/类别/分数、原始点索引、XY凸包和面积
    points.npz                    # 原始xyz/raw labels、6xN camera_instance_ids和camera_names
    summary.json
    per_camera/<camera>_hulls.png
    cameras/<camera>_comparison.jpg      # 原图/实例掩码/分配LiDAR点
    cameras/<camera>_instances_overlay.jpg
    cameras/<camera>_semantic_ids.png
    cameras/<camera>_instances.json
    cameras/<camera>_instances.npz       # packed原始mask、图像实例图、投影点/像素/实例ID
```

原始掩码可用 `np.unpackbits(packed_masks[i], count=H*W, bitorder='little').reshape(H,W)` 解码，H/W来自 `mask_shape_hw`。凸包顶点单位是原始LiDAR XY米坐标，保留输入精度；至少三个非共线支持点才能形成有面积多边形。单点或共线点只保留点/线段，面积为0；无ROI分配点的检测保留空记录，不画成填充物体。

**局限：** 凸包会填满未观测的内部并跨越空隙，它不是新增LiDAR观测、occupancy真值或可靠物体轮廓。跨像素遮挡、mask误差和背景点会明显拉长某些实例凸包。当前不静默去离群点或做几何聚类；下述共享点合并单独记录并可关闭，不会修复被拉长的凸包，也不能证明物理身份。本地lidarseg仅有语义类别，没有与SAM3对应的实例身份，因此不编造实例/凸包精度指标。

### 用共享LiDAR点索引做跨相机合并

`run_instances.sh` 现在默认额外生成 `merged/` 合并前后对比，原掩码、点分配和凸包保留。`--no-merge-overlap` 可关闭关联。默认要求 **至少3个共享原始点**，且 **交集点数 / 较小点集大小 >= 0.30**，可通过 `--min-shared-points` 和 `--min-overlap` 调整。

仅比较不同相机、同类别、非空的前景ID1..10。匹配使用完整scan的原始点索引，不用BEV多边形重叠或GT。逐对记录点集大小、交集、并集、点IoU、较小点集重叠率、阈值是否通过及最终决定；候选按重叠率、IoU、共享点数、较低实例置信度降序确定性处理。每组 **同一相机最多一个检测**，并要求 **所有成员两两通过阈值**（complete-link），避免A–B–C弱证据链自动连成大实例。11..16的stuff以及未匹配/空实例保持独立。

融合组使用最小成员ID及其颜色，点集取并集并去重，分数取成员最大值（不是联合概率），重新计算ROI凸包。不新增点、不静默删除点；不同未关联组仍可能共享点。这是单scan实验，不生成跨帧轨迹身份。

已有实例结果无需重跑SAM3即可合并：

```bash
bash tasks/nuscenes_sam3/scripts/merge_instances.sh \
  --input-dir outputs/nuscenes_instance_hulls_experiment \
  --samples 1 --min-shared-points 3 --min-overlap 0.3 \
  --output-dir outputs/nuscenes_instance_merge_demo
```

独立入口生成 `result.html` 和每帧子目录；自动合并时同样文件位于 `<frame>/merged/`：`merge_comparison.png`（前景合并前后）、`matched_groups.png`（逐组XY放大）、`bev_before_merge.png`、`bev_after_merge.png`、`bev_all_groups.png`、`merged_hulls.json`、`match_pairs.json/csv`、`instance_mapping.json`、`merged_points.npz` 和 `summary.json`。NPZ保留 `camera_instance_ids` 并新增 `camera_merged_ids`，JSON中的组级点索引已去重。关联阈值不是精度保证，背景共享点或错误掩码仍可能造成误关联；前面的凸包局限依旧存在。

## 任务导航

| 任务目录 | 用途 | 示例命令（从项目根目录执行） |
|---|---|---|
| [Sidewalk](../tasks/sidewalk/) | 语义分割与像素评估 | `bash tasks/sidewalk/scripts/segment_sidewalk.sh --help` |
| [nuImages](../tasks/nuimages/) | 已标注图像分割与像素评估 | `bash tasks/nuimages/scripts/segment_nuimages.sh --help` |
| [nuScenes SAM3](../tasks/nuscenes_sam3/) | 相机分割、投影点、融合与二维 BEV | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --help` |
| [EFFOcc](../tasks/effocc/) | Fusion-R18 占用、相机投影与 SAM3 对比 | `bash tasks/effocc/scripts/run_effocc_mini.sh --help` |
| [平地投影](../tasks/flat_ground/) | 纯相机平面语义IPM，不用LiDAR | `bash tasks/flat_ground/scripts/run.sh --help` |
| [共享工具](../shared/) | 环境入口、公共代码、自定义类别与全量测试 | `bash shared/scripts/test_all.sh` |

## 环境配置

以下命令均在 **WSL Bash** 中执行，项目路径为 `/mnt/d/wsl_workplace/vla-supervised-ad`。当前发行版是 Ubuntu-26.04，使用 Conda 环境 **`vla-supervised`**，默认 `cuda + bf16` 推理。本机 NVIDIA RTX 5070 Ti 已完成实际推理验证。

### 使用当前环境

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# 手动激活，便于运行 Python 或检查依赖。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla-supervised
python --version
nvidia-smi
```

当前环境已具备全部运行依赖，日常运行可直接使用下面的 Bash 入口。[run_vla.sh](../shared/scripts/run_vla.sh) 会选择 `vla-supervised`，未手动激活时使用 `conda run -n vla-supervised`，不使用 base 或 Windows Python。解释器当前为 `/home/administrator/miniconda3/envs/vla-supervised/bin/python`。如果 Conda 装在其它位置，先激活该环境，或设置 `CONDA_EXE=/实际安装位置/bin/conda`。

| 组件 | 当前安装版本 / 路径 | 用途 |
|---|---|---|
| Python | 3.12.14 | 项目解释器 |
| torch | 2.14.1+cu132 | GPU 推理 |
| torchvision | 0.29.1+cu132 | SAM3 图像预处理 |
| sam3 | 0.1.0，editable 源码位于 `/mnt/d/wsl_workplace/sam3` | 官方 SAM3 图像模型 |
| NumPy | 2.5.3 | 掩码、投影、融合、指标 |
| Pillow | 12.3.0 | 图像读写和二维可视化 |
| PyArrow | 26.0.0 | Sidewalk Parquet 读取 |
| pycocotools | 2.0.11 | nuImages RLE 掩码解码 |
| matplotlib | 3.11.2 | 现有可选库；当前BEV导出不需要 |
| einops | 0.8.2 | SAM3 内部依赖 |
| setuptools | 80.10.2 | 当前 SAM3 使用的 `pkg_resources` |

这是当前机器的已验证安装记录，不是跨机器环境锁定文件。当前 SAM3 包声明 `numpy<2`，而现有环境为 NumPy 2.5.3，`pip check` 会报告这项版本不一致；本项目各模式的实际推理和回归测试已通过。下面的新建环境步骤按 SAM3 声明安装 NumPy 1.x；无需为了运行这些命令修改现有环境。

### 在另一台 WSL 中新建环境

先运行 `conda env list`。如果已有 `vla-supervised`，直接使用；只有没有该环境时才执行 `conda create`。以下是安装说明，本次没有重新创建或替换现有环境。

```bash
# 在已安装 Miniconda 的 WSL 中执行；路径按实际安装位置调整。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env list

# 仅当 vla-supervised 尚不存在时执行。
conda create -n vla-supervised python=3.12 -y
conda activate vla-supervised

# 选择支持当前 NVIDIA GPU 的成对 torch/torchvision CUDA wheel。
# cu128 与 PyTorch 2.10.0 是本地 SAM3 README 提供的安装示例。
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install "numpy>=1.26,<2" "setuptools<81" einops matplotlib

# 本机已有 SAM3 源码；在新机器上先把官方源码放到这个位置。
# git clone https://github.com/facebookresearch/sam3.git /mnt/d/wsl_workplace/sam3
python -m pip install -e /mnt/d/wsl_workplace/sam3

cd /mnt/d/wsl_workplace/vla-supervised-ad
bash tasks/sidewalk/scripts/setup_sidewalk.sh
bash tasks/nuimages/scripts/setup_nuimages.sh
```

[setup_sidewalk.sh](../tasks/sidewalk/scripts/setup_sidewalk.sh) 根据 [requirements.txt](../tasks/sidewalk/requirements.txt) 补充 PyArrow、Pillow、NumPy；[setup_nuimages.sh](../tasks/nuimages/scripts/setup_nuimages.sh) 根据 [requirements.txt](../tasks/nuimages/requirements.txt) 补充 pycocotools、Pillow、NumPy。这两个脚本不会安装 SAM3、torch 或 torchvision。nuScenes 数据读取与投影使用 NumPy/Pillow，不需要 `nuscenes-devkit`；BEV绘图仅使用NumPy/Pillow，不需要WebGL或matplotlib。

模型使用本地官方 `models/sam3/sam3.pt`。模型目录和 SAM3 源码目录分别是 `/mnt/d/model_zoo/sam3` 与 `/mnt/d/wsl_workplace/sam3`：权重目录不会自动提供 `sam3` Python 包。模型构建明确设置 `load_from_HF=False`，无需在运行分割时下载权重。

### 检查环境与数据

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# 输出实际解释器、GPU 可用性、关键库及 SAM3 源码位置；不加载模型。
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

# 检查链接。所有源路径须存在；已有链接不需要重新创建。
ls -ld models/sam3 datasets/sidewalk-semantic datasets/nuscenes datasets/nuimages
readlink -f models/sam3 datasets/sidewalk-semantic datasets/nuscenes datasets/nuimages

# 只读取数据/标注，不运行 SAM3。
bash tasks/sidewalk/scripts/segment_sidewalk.sh --inspect-data
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --inspect-data
bash tasks/nuimages/scripts/segment_nuimages.sh --inspect-data
```

环境检查中 CUDA 不可用时，先在 WSL 中核对 `nvidia-smi` 和 torch 的 CUDA wheel。所有分割入口也支持 `--device cpu --precision fp32`，CPU 运行速度会慢于 GPU。

## 可用命令总表

以下命令均假定已进入项目根目录。每次分割或点云导出使用**新的或空的输出目录**。表中示例均限制样本数，不会启动全量推理；完整参数可对分割/导出入口追加 `--help`。

| 用途 | 可直接执行的命令 | 输出 / 默认行为 |
|---|---|---|
| Sidewalk 数据集类别分割与评估 | `bash tasks/sidewalk/scripts/segment_sidewalk.sh --max-images 5 --output-dir outputs/sidewalk_demo` | 预测、GT 对比可视化、像素指标；默认 train |
| Sidewalk TXT 自定义类别 | `bash tasks/sidewalk/scripts/segment_custom.sh --classes-file shared/config/classes.txt --max-images 5 --output-dir outputs/sidewalk_custom` | 预测与可视化，不计算 GT 指标 |
| nuScenes 前视相机，16 类 | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --max-images 5 --output-dir outputs/nuscenes_demo` | 默认 mini_val / CAM_FRONT；lidarseg 投影点观测评估 |
| nuScenes 原始 32 类 | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --taxonomy lidarseg32 --max-images 5 --output-dir outputs/nuscenes_raw32` | 保留原始 lidarseg IDs |
| nuScenes 六相机 + 二维 BEV | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --bev-resolution 0.2 --output-dir outputs/nuscenes_six_demo` | 默认 6 张相机图＝1 帧 LiDAR，自动融合、导出二维 BEV 和唯一点指标 |
| nuScenes 六相机处理 10 帧 | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --max-images 60 --bev-resolution 0.2 --output-dir outputs/nuscenes_six_10frames` | 60 张图像＝10 帧，各帧独立融合 |
| nuScenes TXT 自定义类别 | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_custom.sh --classes-file shared/config/classes.txt --max-images 5 --output-dir outputs/nuscenes_custom` | 图像分割与可视化，不计算类别 GT 指标 |
| nuScenes 六相机 TXT 自定义类别 + 二维 BEV | `bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --mode custom --classes-file shared/config/classes.txt --bev-resolution 0.2 --output-dir outputs/nuscenes_six_custom` | 预测/覆盖BEV和完整NPZ，不计算lidarseg语义指标 |
| 已有 nuScenes 分割结果染色到点云 | `bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --bev-resolution 0.2 --output-dir outputs/nuscenes_demo_cloud` | 无需重跑SAM3；二维BEV PNG/栅格/元数据、完整NPZ和数据集模式唯一点评估 |
| 仅导出已有结果中的 1 帧点云 | `bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --max-frames 1 --bev-resolution 0.2 --output-dir outputs/nuscenes_first_cloud` | 数量单位为 LiDAR keyframe |
| nuImages 类别分割、GT 比较与评估 | `bash tasks/nuimages/scripts/segment_nuimages.sh --max-images 5 --output-dir outputs/nuimages_demo` | 默认 v1.0-mini，24 个非 ego 类别；六栏对比、像素指标 |
| nuImages 仅筛选前视图 | `bash tasks/nuimages/scripts/segment_nuimages.sh --cameras CAM_FRONT --max-images 5 --output-dir outputs/nuimages_front` | 只选已标注 key camera 图像 |
| nuImages 同时提示和评估 ego | `bash tasks/nuimages/scripts/segment_nuimages.sh --include-ego --max-images 5 --output-dir outputs/nuimages_with_ego` | 包含 ID31，共25个提示类别 |
| nuImages TXT 自定义类别 | `bash tasks/nuimages/scripts/segment_nuimages_custom.sh --classes-file shared/config/classes.txt --max-images 5 --output-dir outputs/nuimages_custom` | 图像分割与可视化，不计算 GT 指标 |
| 建立/核对 nuImages 软链接 | `bash tasks/nuimages/scripts/link_nuimages.sh` | 指向 `/mnt/d/datasets/nuimages`，已有不同目标时不覆盖 |
| 补充 Sidewalk 数据读取依赖 | `bash tasks/sidewalk/scripts/setup_sidewalk.sh` | 固定安装到 vla-supervised |
| 补充 nuImages 掩码解码依赖 | `bash tasks/nuimages/scripts/setup_nuimages.sh` | 固定安装到 vla-supervised |
| 全部回归测试 | `bash shared/scripts/test_all.sh` | 执行全部任务及共享回归测试 |
| nuScenes 数据/导出管线测试 | `bash tasks/nuscenes_sam3/scripts/test_nuscenes.sh` | 覆盖 nuScenes SAM3 本任务全部测试，含 BEV、fusion 和 viewer |
| nuImages 数据/管线测试 | `bash tasks/nuimages/scripts/test_nuimages.sh` | 覆盖 nuImages 本任务全部测试 |
| EFFOcc Fusion-R18 mini推理 | `bash tasks/effocc/scripts/run_effocc_mini.sh --max-samples 1 --output-dir outputs/effocc_demo` | 六相机加LiDAR/sweeps；原生200x200x16占用与二维BEV |
| 同帧 SAM3 / EFFOcc 对比 | `bash tasks/effocc/scripts/compare_effocc_sam3.sh --input-dir outputs/effocc_mini_with_cameras --max-samples 1 --output-dir outputs/effocc_sam3_demo` | 六路相同相机帧的原图、SAM3和occupancy叠加 |
| EFFOcc occupancy投影到六相机 | `bash tasks/effocc/scripts/project_effocc_cameras.sh --input-dir outputs/effocc_mini_verified --max-samples 1 --output-dir outputs/effocc_camera_demo` | 复用已保存预测；原尺寸叠加、渲染类别/深度和六视角拼图 |
| EFFOcc 输入检查 | `bash tasks/effocc/scripts/run_effocc_mini.sh --inspect-data` | 只检查mini输入数组，不运行模型 |
| EFFOcc 适配测试 | `bash tasks/effocc/scripts/test_effocc.sh` | 稀疏算子、预处理、pooling和输出测试 |
| 在固定环境执行 Python 命令 | `bash shared/scripts/run_vla.sh -m pip list` | 通用环境入口，可传 Python 的任意参数 |

### 数量、参数与查看结果

| 参数 / 默认值 | Sidewalk | nuScenes 通用入口 | nuScenes 六相机入口 | nuImages |
|---|---|---|---|---|
| 默认处理数量 | 全部剩余图像 | 10 张相机图 | 6 张相机图＝1帧 | 10 张标注图 |
| `--max-images N` 单位 | 图像 | 相机图像 | 相机图像，建议6的倍数 | 筛选后的标注图 |
| `--max-images 0` | 全部剩余 | 全部选定视角 | 全部选定六视角 | 全部剩余 keyframes |
| 数据范围 | `--split train/validation/test` | `--split all/mini_train/mini_val` | 同 nuScenes | `--version v1.0-mini` / `v1.0-train` / `v1.0-val` / `v1.0-test` |
| 类别模式 | 数据集 / custom | `lidarseg16/lidarseg32` / custom | 同 nuScenes | 数据集 / custom；可含 ego |

三个数据集的分割入口都支持 `--start-index`、`--classes-file`、`--score-threshold`、`--mask-threshold`、`--device`、`--precision`、`--alpha`、`--vis-every`、`--no-vis`、`--inspect-data`、`--output-dir`。默认阈值均为0.5，默认 `cuda/bf16`。模型和数据路径可通过 `--checkpoint`、`--dataset-dir` 替换。不同 split/version 必须实际存在；本地 Sidewalk 只有 train，nuScenes 和 nuImages 只有 mini。

nuScenes 使用 `--cameras`、`--scenes`、`--min-depth`、`--export-pointcloud`；nuImages 的 `--cameras` 是筛选标注 key camera，不代表每个 sample 都有六张标注。BEV独立导出入口支持 `--max-frames`、`--bev-resolution`（默认0.2米/像素）、`--bev-density-cap`（默认胜出类别32点达到最深色）、`--no-preview`。分割入口也支持这两个BEV参数，它们只改变显示；NPZ和点级指标保留全部点。

- 二维对比图：各分割输出的 `visualizations/`；原始类别图位于 `predictions/`。
- 图像指标 / 相机投影点观测指标：分割输出根目录的 `metrics.json`、`metrics.csv`。
- 六相机BEV：`pointcloud_results/pointclouds/<scene_frame_sample>/` 包含二维PNG、栅格数组、元数据和完整原始点NPZ。
- 融合后的唯一点指标：点云输出的 `evaluation/metrics.json`，分别报告 covered_points、full_scan 和 coverage，口径不同于分割输出根目录的投影观测指标。
- TXT 自定义模式不输出语义 GT 评估，nuScenes 自定义点云仍会报告几何覆盖率。

各模式的类别映射、输出字段和评估细节见后续章节。

## 项目目录与数据路径

仅支持 **WSL + Conda `vla-supervised`**。启动脚本会选择该环境，不使用 base 或 Windows Python。每个任务集中管理 Python 实现、Bash 入口、测试，以及任务专用类别配置或依赖；公共运行时、SAM3 支持、指标和 nuScenes 数据工具放在 [shared/python/](../shared/python/)。

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
  config/classes.txt          # 所有任务共用的 TXT 自定义类别
  scripts/                    # run_vla.sh, test_all.sh
python/                       # 旧 Python 兼容转发入口
scripts/                      # 旧 Bash 兼容转发入口，含 test_all.sh
config/                       # 旧类别配置兼容副本
tests/                        # 旧测试兼容 import，实际实现位于任务/共享目录
requirements-*.txt            # 旧依赖别名：-r tasks/<task>/requirements.txt
datasets/                     # 保留数据与软链接
models/                       # 保留模型源码与 SAM3 权重软链接
assets/                       # 保留 checkpoint
outputs/                      # 保留新旧结果，默认运行目录名称不变
```

激活环境后，规范 Python 模块入口须**从项目根目录执行**：

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad
python -m tasks.sidewalk.python.sam3_sidewalk --help
python -m tasks.nuimages.python.sam3_nuimages --help
python -m tasks.nuscenes_sam3.python.sam3_nuscenes --help
python -m tasks.effocc.python.effocc_mini --help

# 无需手动激活即可选择指定环境。
bash shared/scripts/run_vla.sh -m tasks.nuscenes_sam3.python.sam3_nuscenes --help
```

任务 Bash 入口自动切换到项目根目录，再通过 [shared/scripts/run_vla.sh](../shared/scripts/run_vla.sh) 执行 `-m tasks.<task>.python.<module>`；脚本原名和 CLI 参数保持不变。公共 [runtime.py](../shared/python/runtime.py) 和 [sam3.py](../shared/python/sam3.py) 提供运行及模型支持，[segmentation_metrics.py](../shared/python/segmentation_metrics.py) 提供指标，[nuscenes_data.py](../shared/python/nuscenes_data.py) 为 SAM3 与 EFFOcc 提供 nuScenes 数据/投影工具。

[Sidewalk 测试](../tasks/sidewalk/scripts/test_sidewalk.sh) 只运行 Sidewalk 测试；[nuImages 测试](../tasks/nuimages/scripts/test_nuimages.sh) 只运行 nuImages 测试；[nuScenes SAM3 测试](../tasks/nuscenes_sam3/scripts/test_nuscenes.sh) 覆盖本任务全部测试，含 BEV、fusion 和 viewer；[EFFOcc 测试](../tasks/effocc/scripts/test_effocc.sh) 覆盖其适配器。全量测试使用 [shared/scripts/test_all.sh](../shared/scripts/test_all.sh)。旧 [scripts/test_sidewalk.sh](../scripts/test_sidewalk.sh) 仍执行全部测试，新增 [scripts/test_all.sh](../scripts/test_all.sh) 作为兼容入口；根目录测试模块保留兼容 import，指向实际任务/共享实现。

原顶层 Python 和 Bash 路径继续作为兼容转发入口，顶层类别 TXT 保留兼容副本。三个旧 requirements 文件通过 `-r tasks/<task>/requirements.txt` 转发；推荐使用上面的任务依赖和类别配置，所有自定义类别模式共用 [shared/config/classes.txt](../shared/config/classes.txt)。EFFOcc/SAM3 对比默认复用 [nuScenes 16 类配置](../tasks/nuscenes_sam3/config/nuscenes_classes.txt)。本次整理改变导入布局和入口位置，无需重装环境；数据、模型源码、checkpoint 和输出均不移动，模型权重未改，新旧输出默认目录名称保持原样。后文推理和验证记录来自已有运行，不代表目录整理后新增的推理结果。

项目使用以下已有 WSL 符号链接：

- `models/sam3` → `/mnt/d/model_zoo/sam3`，使用官方 `sam3.pt`。
- `datasets/sidewalk-semantic` → `/mnt/d/datasets/sidewalk-semantic`，直接读取 `data/train-*.parquet` 和 `id2label.json`。
- `datasets/nuscenes` → `/mnt/d/datasets/nuscenes`，使用 `v1.0-mini/`、`samples/` 和 `lidarseg/v1.0-mini/`。
- `datasets/nuimages` → `/mnt/d/datasets/nuimages`，使用 `v1.0-mini/` 与 `samples/` 的二维标注图像。

当前环境的激活、依赖补充和新建环境步骤见前面的[环境配置](#环境配置)。下面按数据集介绍类别配置、输出和评估。

## 1. 使用 Sidewalk Semantic 类别并评估

先用少量图像确认输出：

```bash
bash tasks/sidewalk/scripts/segment_sidewalk.sh \
  --max-images 5 \
  --output-dir outputs/sidewalk_demo
```

评估全部 1000 张训练集图像：

```bash
bash tasks/sidewalk/scripts/segment_sidewalk.sh \
  --output-dir outputs/sidewalk_full
```

默认读取原始 34 个非零类别，保留数据集 ID 1–34；ID 0 为 `unlabeled`。[sidewalk_classes.txt](../tasks/sidewalk/config/sidewalk_classes.txt) 将 `flat-sidewalk` 等数据集标签名映射到 `sidewalk` 等更适合 SAM3 的英文提示词。可以修改 `=` 右侧的提示词、用 `|` 添加别名；左侧必须保留完整的原始类别名。也可以使用 `--classes-file tasks/sidewalk/config/your_sidewalk_prompts.txt` 指定完整的替代映射。

原始类别包含 `void-ground`、`void-dynamic`、`void-static`、`void-unclear`。这些标签缺乏精确的自然语言视觉概念，配置中使用近似提示词，但评估仍按原始 ID 计分。这是 SAM3 文本提示的零样本语义分割评估，没有训练语义分类头；提示词、置信度阈值和掩码冲突处理均会影响指标。

本地数据仅有 `train` 划分。脚本也接受 `--split validation/test`，需要用户另行提供同格式的对应 Parquet 文件。按子集运行时，指标只覆盖实际处理的图像，不代表整个数据集。

## 2. 使用 TXT 自定义类别

编辑 [classes.txt](../shared/config/classes.txt)，每行一个英文类别：

```text
road
sidewalk
person
car
```

支持单独设置显示类别名和一个或多个英文提示词：

```text
plants = vegetation | tree | bush
crosswalk = pedestrian crosswalk
```

支持空行、`#` 注释及 UTF-8 BOM。自定义类别按配置顺序分配 ID 1、2、3……，ID 0 表示未覆盖像素；最多 254 个类别。同名类别、空提示词和保留名称 `unlabeled`/`unassigned` 会报错。

```bash
bash tasks/sidewalk/scripts/segment_custom.sh \
  --classes-file shared/config/classes.txt \
  --max-images 5 \
  --output-dir outputs/custom_demo
```

自定义类别模式仍对 Sidewalk 图像分割，但不读取真值标签、不计算语义指标，因为自定义类别与原始标签可能不对应。

## 输出结果

每次运行使用一个新的或空的 `--output-dir`，避免混合不同参数的结果；不会覆盖已有非空输出目录。

```text
outputs/<run>/
  predictions/train_000000.png     # 原始尺寸、单通道 uint8 类别 ID 图
  color_masks/train_000000.png     # 彩色语义分割图
  overlays/train_000000.jpg        # 原始尺寸图像叠加
  visualizations/train_000000.jpg  # 原图/预测叠加/预测类别图/GT（仅数据集模式）对比及图例
  instances/train_000000.json      # 提示词、类别、实例分数、框及合并前掩码像素数
  classes.json                    # ID、类别名、RGB 颜色、提示词
  run_config.json                 # 参数、环境、数据范围、运行状态及时间
  samples.jsonl                   # 每张图像的索引、尺寸、实例数、覆盖率及处理时间
  metrics.json                    # 仅数据集类别模式：总体和每类指标
  metrics.csv                     # 仅数据集类别模式：总体和每类指标表
  confusion_matrix.csv            # 仅数据集类别模式：GT 行、预测列
```

`instances` 保存实例元数据，最终像素级分割保存于 `predictions`。可视化对比图会按单面板最大宽度 640 缩小展示，预测掩码和叠加图保持原始尺寸。ID 0 在彩色图中为黑色，在叠加图中保留原图；GT 255 在图例中标记为 ignored。

SAM3 每张图像只编码一次，逐类别/提示词生成实例掩码。先按 `--score-threshold` 筛选实例，再按 `--mask-threshold` 对像素概率二值化。同一像素被多个实例覆盖时，采用实例置信度最高者的类别；分数相同时保留较早的类别/提示词/实例。未覆盖像素设为 0。同一类别的多个实例自然合并为该类别。

## 评估口径

- 忽略 **GT ID 0 和 255**，这些像素完全不进入混淆矩阵。
- 在其它有效 GT 像素上，预测 ID 0 仍进入混淆矩阵，计为真实类别的假阴性，不因“未分割到”而从评估中删除。
- 按整个所选数据子集累计像素混淆矩阵，再计算指标，不直接平均各图的指标。
- 输出 `pixel_accuracy`（像素准确率）、`mean_iou`（mIoU）、`mean_accuracy`（平均类别召回率）、`frequency_weighted_iou`（频率加权 IoU）、`macro_dice`，以及每类 IoU、Dice、precision、recall、TP、FP、FN 和像素数。
- 类别 0 从宏指标排除。mIoU/Dice 排除 GT 和预测都未出现的类别；只有误报、没有 GT 的类别仍以 0 分计入。平均类别准确率只包含有 GT 的类别。
- 指标使用 **0–1** 数值。没有可定义分母的指标写为 JSON `null` / CSV 空单元格，不写 NaN。JSON 还包含公式和忽略策略。

异常或 Ctrl+C 后，如果已进入样本处理阶段，会保存当前累计指标，并在 `run_config.json` 标记失败/中断；此时结果是部分处理结果，应查看 `status` 和 `processed_images`。

## 常用参数与检查

```bash
# 只检查 Parquet 字段、样本尺寸和标签 ID，不加载 SAM3
bash tasks/sidewalk/scripts/segment_sidewalk.sh --inspect-data

# 从全局索引 100 开始处理 10 张，降低输出可视化数量
bash tasks/sidewalk/scripts/segment_sidewalk.sh \
  --start-index 100 --max-images 10 --vis-every 5 \
  --score-threshold 0.5 --mask-threshold 0.5 \
  --output-dir outputs/sidewalk_subset

# 查看完整参数
bash tasks/sidewalk/scripts/segment_sidewalk.sh --help

# 全部回归测试，含类别配置、掩码合并、数据读取、可视化和评估指标
bash shared/scripts/test_all.sh
```

默认 `--device cuda --precision bf16`，适合本机 RTX 5070 Ti。支持 `--precision fp32/fp16`；CPU 模式需同时指定 `--device cpu --precision fp32`。`--no-vis` 只保存类别 ID 图、实例信息和评估结果。`--max-images 0` 表示处理从起始索引到结尾的全部图像。

## 3. nuScenes mini：相机分割与 lidarseg 投影点评估

仍然仅在 WSL 的 `vla-supervised` 环境运行，复用本地 SAM3。读取 JSON 与点云投影使用 NumPy/Pillow，不需要安装 `nuscenes-devkit` 或修改已有环境。

默认使用 `mini_val` 的 `CAM_FRONT`，**最多处理 10 张图像**，类别使用官方 32→16 类合并。你可以先限制为 1–5 张：

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --max-images 5 \
  --output-dir outputs/nuscenes_demo
```

选择 mini 全集中的一个场景和多个相机：

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --split all --scenes scene-0061 \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
  --max-images 6 \
  --output-dir outputs/nuscenes_scene0061
```

`--split` 支持 `mini_train`、`mini_val`、`all`。场景筛选与 split 取交集。六相机可选 `CAM_FRONT`、`CAM_FRONT_LEFT`、`CAM_FRONT_RIGHT`、`CAM_BACK`、`CAM_BACK_LEFT`、`CAM_BACK_RIGHT`。只使用 keyframe，与同一个 sample 的 `LIDAR_TOP` 和 lidarseg 标签关联，不使用 sweeps。按场景名、sample 时间戳、指定相机顺序枚举。

本地 mini 共 404 个 sample：全六相机为 2,424 张图像；mini_train 为 323 个 sample，mini_val 为 81 个 sample。**`--max-images` 和 `--start-index` 的单位是选定列表中的相机图像**，例如六相机配置下 `--max-images 6` 处理一个 sample 的六个视角。`--max-images 0` 才处理从起始索引到结尾的全部选定视角。每次运行需使用新的或空的输出目录。

### 类别模式

- 默认 `--taxonomy lidarseg16`：16 个有效类别，ID 1–16；ID 0 为 ignore/未覆盖。使用 [nuscenes_classes.txt](../tasks/nuscenes_sam3/config/nuscenes_classes.txt)，保留左侧类别名，右侧可修改提示词或添加 `|` 别名。类别为 barrier、bicycle、bus、car、construction_vehicle、motorcycle、pedestrian、traffic_cone、trailer、truck、driveable_surface、other_flat、sidewalk、terrain、manmade、vegetation。
- `--taxonomy lidarseg32`：保留原始标签 ID 0–31，0=noise 被忽略，1–31 全部提示并评估。使用 [nuscenes_lidarseg32_classes.txt](../tasks/nuscenes_sam3/config/nuscenes_lidarseg32_classes.txt)。人群细分类及 `vehicle.ego` 等文本提示存在语义歧义；此模式是原始类别实验，官方 benchmark 通常使用合并后的 16 类。
- `--classes-file` 可以替换该 taxonomy 的完整提示配置；类别名与 IDs 不改变。

```bash
# 原始 lidarseg 32 类
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --taxonomy lidarseg32 --max-images 5 \
  --output-dir outputs/nuscenes_raw32

# 自定义 TXT 类别，仅分割和可视化，不计算 lidarseg 指标
bash tasks/nuscenes_sam3/scripts/segment_nuscenes_custom.sh \
  --classes-file shared/config/classes.txt --max-images 5 \
  --output-dir outputs/nuscenes_custom
```

类别映射按官方源码核对并写入每次评估结果的 `lidarseg_mapping.json`。本地 `category.json` 是检测框类别表，缺少 lidarseg 索引，脚本不会用该表推断标签。官方参考：[原始类别/索引顺序](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py)、[32→16 合并](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/utils.py)、[mini split 场景列表](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py)。

### 投影与评估口径

lidarseg 每个标签对应一个 LiDAR 点，不是整张相机图像的像素标注。脚本将点云按以下链投影到相机：

```text
LiDAR sensor -> LiDAR 时刻 ego -> global
             -> camera 时刻 ego -> camera sensor -> camera intrinsic
```

使用各传感器时间戳对应的 ego pose，四元数按 `w,x,y,z` 解读。保留 camera 深度大于 `--min-depth`（默认 1 m）、位于 1 像素内边界内的点。以 `floor(u), floor(v)` 采样全分辨率预测图；同一整数像素的多个投影点只保留最近深度点，深度相同保留原始点索引较早者。标签和点数量必须一致，原始标签需在 0–31，否则报错。

- 16 类模式按官方映射忽略 GT coarse ID 0，包括原始 noise、ego 和其它官方排除类别；32 类模式只忽略 GT raw ID 0。
- 有效 GT 点上的预测 ID 0 仍按错误计数，不能丢弃未覆盖的点。
- 按所有处理视角累计点观测混淆矩阵，输出点准确率、mIoU、平均类别召回率、频率加权 IoU、macro Dice 和逐类指标。宏指标处理规则与 Sidewalk 相同。
- **这是相机中的稀疏投影点评估**，既不是密集图像语义分割指标，也不是官方全方位点云 benchmark 分数。跨像素遮挡以及运动物体的传感器时间差没有完全消除；同一个 LiDAR 点落在多个相机时按每个相机观测分别计数。
- `metrics.json` 包含 `point_accuracy`、`valid_point_observations` 和详细的 `evaluation_scope`。为兼容共享指标实现，也保留 `pixel_accuracy/valid_pixels` 及逐类 `gt_pixels/predicted_pixels` 别名，但此模式单位全部是**投影点观测**。CSV 使用 `point_accuracy`、`gt_points`、`predicted_points`，每行标明单位。
- 自定义模式没有标签对应关系，不生成评估指标或投影 GT；本地 mini 数据目录仍需完整的 lidarseg 元数据，供 keyframe 关联检查。

### nuScenes 输出

与 Sidewalk 同样生成 `predictions/`、`instances/`、`overlays/`、`color_masks/`、`visualizations/`、`classes.json`、`run_config.json`、`samples.jsonl`。文件名前缀包含场景、选定视角索引、相机和 sample token。额外输出：

```text
projected_points/<stem>.npz      # point_indices、uv、pixels、depth、raw_labels、target、prediction
sparse_gt/<stem>.png            # uint8 原始尺寸稀疏 GT，未标注像素为 255
point_overlays/<stem>_gt.jpg     # 彩色 lidarseg GT 点叠加在原图
point_overlays/<stem>_prediction.jpg # 相同投影点的预测类别叠加
lidarseg_mapping.json           # 每个 raw ID 到当前 taxonomy 的映射
metrics.json / metrics.csv / confusion_matrix.csv
```

对比图显示原图、SAM3 叠加、密集预测类别图、**稀疏 LiDAR GT 点叠加**，不会用稀疏标注填充成密集 GT。`--no-vis` 关闭可视化，但仍保存类别图、投影点、稀疏 GT 与评估。异常/中断后检查 `run_config.json` 的 `status/processed_images`，部分累计指标不能当作完整运行。

```bash
# 不加载模型，检查选定图像和真实投影点数量
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --inspect-data

# 查看全部参数
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh --help

# nuScenes SAM3 全部测试，含投影、指标、BEV、fusion 和 viewer
bash tasks/nuscenes_sam3/scripts/test_nuscenes.sh
```

## 4. 原始 LiDAR 点回填与二维 BEV 导出

像素预测通过同一套标定投影回填到**原始点索引**，再融合已完成的相机视角。染色点云投影到 XY 鸟瞰图（BEV）栅格，坐标保持原生 `LIDAR_TOP` 传感器坐标系，单位米；**+x 向右、+y 向上**，不假定车辆前方朝上。原始点数组和点级评估保留整帧全量点。

### 已有分割结果直接导出，无需重新推理

```bash
bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh \
  --input-dir outputs/nuscenes_demo \
  --bev-resolution 0.2 \
  --output-dir outputs/nuscenes_demo_cloud
```

输入为已完成的 nuScenes 分割输出目录，脚本读取 `samples.jsonl`、`predictions/` 和配置。也可以从失败/中断的运行导出已记录完成的图像，来源状态会保留在输出中；不要在源分割任务仍运行时导出。自定义类别模式也可导出预测点云和覆盖率，但不计算与 lidarseg 的类别指标。

`--max-frames 1` 只导出一个 LiDAR keyframe；默认 0 导出输入中全部已完成的点云帧。`--dataset-dir` 可以替换源配置中的数据根目录。使用新的或空的输出目录，避免混入不同推理结果。

### 一次运行六相机分割、染色和评估

六视角专用入口默认处理一个 keyframe（6 张相机图），自动导出二维BEV、完整NPZ和指标：

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh --bev-resolution 0.2 --output-dir outputs/nuscenes_six_demo
```

若处理 10 帧，指定 `--max-images 60`；建议 `--max-images` 和 `--start-index` 使用 6 的倍数，避免首尾帧只有部分视角。也可使用通用入口：

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes.sh \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \
  --max-images 6 \
  --export-pointcloud \
  --bev-resolution 0.2 \
  --output-dir outputs/nuscenes_one_scan
```

这里 6 张图像对应一个 sample 的六个相机，最终融合成一帧彩色点云。默认 `--split mini_val --taxonomy lidarseg16`，可使用原始 `lidarseg32` 或自定义类别入口。`--export-pointcloud` 接在原来的分割命令上即可，自动导出到 `<output-dir>/pointcloud_results/`。已有单相机结果只包含该相机覆盖的区域；点云导出不会自动运行其它相机。相机列表被 `--start-index`/`--max-images` 截断时，该帧仅融合实际完成的视角，输出中会记录相机名单。

同一个原始点在多个相机中出现时，采用**非零类别多数票**；0 表示未分割到，作为弃票，但点仍标记为几何覆盖。平票时优先输入清单中较早的相机。每个点在融合后只评估一次，避免重复计算相机重叠区域。融合不读取 GT，不用真值挑选预测。

### BEV图像与原始点数组

```text
<pointcloud-output>/
  run_config.json / frames.jsonl / classes.json
  evaluation/
    metrics.json / metrics.csv / coverage.csv
    confusion_matrix_covered.csv / confusion_matrix_full_scan.csv
  pointclouds/<scene_frame_sample>/
    points.npz                # 全量原始点顺序、类别、覆盖和融合来源
    bev_raster.npz            # 多数类别、总点数/胜出类点数/覆盖点数及稀疏类别直方图
    bev_metadata.json         # 分辨率、XY范围、计数规则和颜色深浅刻度
    bev_prediction.png        # 多数预测类别，按该类点数染色深浅
    bev_ground_truth.png      # 独立投票GT类别，按GT胜出类别点数染色
    bev_errors.png            # 格子多数类别对比，不是原始点级指标
    bev_coverage.png          # 有覆盖点的格子，按覆盖点数加深
    bev_density.png           # 每格总点数的灰度密度图
    bev_comparison.png        # 有GT时2x2及深浅图例，自定义模式双面板
    summary.json / evaluation/
```

自动导出位于 `<output-dir>/pointcloud_results/`，独立导出位于指定输出根目录。现在默认 **100米 x 100米固定语义网格、每格0.2米 x 0.2米，对应500 x 500像素**。保留原有LiDAR坐标和语义类别，X/Y范围均为 `[-50, 50)` 米，方向仍为 **+x向右、+y向上**。`--bev-roi-size 100` 控制正方形边长，`--bev-resolution 0.2` 控制网格粒度，边长除以粒度须为整数。`--bev-roi-size 0` 可恢复此前按全点云边界自动确定范围。只有ROI内点参与显示网格投票；ROI外点仍保留在完整 `points.npz` 和点级指标中。元数据记录 `roi_input_point_count`、ROI内 `point_count`、`roi_outside_point_count` 和 `grid_mode`。不增加可行驶/障碍物/未知重分类。

每个原始唯一点在其XY格内为自身类别投一票。**点数最多的类别作为格子类别，平票取最小类别ID。** 预测与GT在相同网格内分别独立投票。类别0代表未分类/忽略，也参与投票，不会为了偏向非零预测而丢弃。高度Z和点顺序不影响格子类别；先做跨相机点融合，因此同一点被多相机看到也只计一次。

**胜出类别的点越多，颜色越深；没有点的格子为纯白色。** 按绝对点数采用对数刻度，将白色混合至类别原色：`n > 0` 时 `alpha = 0.25 + 0.75 * log1p(min(n, cap)) / log1p(cap)`，`RGB = round(255 * (1 - alpha) + base_RGB * alpha)`。默认 `--bev-density-cap 32`，32个或更多支持点达到最深原色。它表示点数，不是置信度/概率，不按每帧归一化。预测和GT各用自己的胜出类别点数，对比图包含1/2/4/8/16/32点颜色示例。

未分类0为灰色，同样按点数加深。覆盖定义为格内至少一个原始点被相机覆盖，蓝色深浅使用 `covered_count`；`bev_density.png` 显示每格总点数。错误图比较 **格子多数类别**，只对有覆盖点且GT多数类别非零的格子显示绿色（类别相同）或红色（不同），忽略格灰色；它与图头给出的全量原始点指标口径不同。

`bev_raster.npz` 包含HxW的 `occupied/covered` 布尔数组、`prediction` 和可选 `target` uint8类别（空格255），以及uint32的 `point_count`、`prediction_count`、可选 `target_count`、`covered_count`，空格计数为0。稀疏数组 `prediction_vote_cells/prediction_vote_classes/prediction_vote_counts` 保留各格各类直方图，有GT时还保存 `target_vote_*`。cell ID为按行展开索引。原来的单点代表 `point_index` 字段移除，因为格子改为聚合全部点。

`points.npz` 按原始顺序完整保留 `xyz/intensity/ring/raw_labels/point_indices/prediction/covered/observation_count/winning_votes/source_camera/camera_names`，数据集类别模式还包含 `target`。**点级评估仍使用所有原始点。** 网格粒度和颜色刻度不改变这些指标，本次不增加BEV像素指标；不插值、不填空格。

单帧六相机入口：

```bash
bash tasks/nuscenes_sam3/scripts/segment_nuscenes_6cams.sh \
  --bev-roi-size 100 --bev-resolution 0.2 --bev-density-cap 32 \
  --output-dir outputs/nuscenes_single_frame_grid
```

已有六相机单帧分割结果可直接重用，无需重跑SAM3：

```bash
bash tasks/nuscenes_sam3/scripts/color_nuscenes_pointcloud.sh \
  --input-dir outputs/smoke_nuscenes_six_cloud --max-frames 1 \
  --bev-roi-size 100 --bev-resolution 0.2 --bev-density-cap 32 \
  --output-dir outputs/nuscenes_single_frame_majority
```

### 点云评估口径

- `covered_points`：仅评估至少一个已处理相机有几何投影覆盖的唯一点，忽略 GT0。已覆盖但预测0仍为错误，不能通过只选成功染色点抬高指标。
- `full_scan`：整帧所有非零 GT 点都计入，未覆盖点预测0作为错误。它反映实际填充整帧的能力；单相机时该分数通常明显较低。
- `coverage`：报告总点数、覆盖点数、实际有非零预测点数、有效 GT 覆盖率及逐类别覆盖率。每帧和全部已导出帧都有指标；GT 合并策略仍由源配置中的 16/32 类模式决定。
- 两种范围都输出 point accuracy、mIoU、mean accuracy、frequency-weighted IoU、macro Dice 和逐类 precision/recall/IoU/TP/FP/FN。0预测造成对应GT的FN；GT0完全排除宏指标和混淆矩阵。多相机同一点只计一次，每帧点索引独立。
- 标定、跨像素遮挡和动态物体时间差仍有误差。点云结果来自相机预测，不是训练的3D语义分割网络输出，也不是官方 nuScenes benchmark 提交分数。

`--no-preview` 关闭全部BEV PNG，保留点/栅格NPZ、元数据、汇总和指标。`--export-pointcloud --no-vis` 有相同效果。迁移说明：已移除 `--html-max-points` 和 `--no-html`，不再生成HTML、PLY或三维预览；已有历史输出不删除。点云指标位于点云输出的 `evaluation/`，原分割输出根目录的 `metrics.json` 仍按每相机投影点观测计数，二者口径不同。

## 5. nuImages：图像分割、真值对比、评估与可视化

本地数据为 `D:\datasets\nuimages` 的 **v1.0-mini，50 张已标注 keyframe**。已创建 WSL 软链接 `datasets/nuimages -> /mnt/d/datasets/nuimages`；可重复执行下面的命令核对/建立链接，已有不同链接或目录时不会覆盖：

```bash
bash tasks/nuimages/scripts/link_nuimages.sh
```

当前 WSL `vla-supervised` 已具备 SAM3、PyTorch、Pillow、NumPy 和 RLE 解码用的 `pycocotools`，无需另装 devkit。另一个同名环境若缺少解码依赖，可执行 `bash tasks/nuimages/scripts/setup_nuimages.sh`，补充项记录在 [requirements.txt](../tasks/nuimages/requirements.txt)。

### 数据集类别模式

```bash
bash tasks/nuimages/scripts/segment_nuimages.sh \
  --max-images 5 \
  --output-dir outputs/nuimages_demo
```

默认 `--version v1.0-mini`、最多 10 张图，选择全部相机通道中的**已标注 key camera**。本地 mini 的 keyframe 分布为前8、前左5、前右7、后11、后左10、后右9。nuImages 每个 sample 提供一张标注 key camera，以及该相机前后的未标注 sweeps；这与 nuScenes 一个 sample 的六个同步相机不同。脚本不会把 sweeps 当作有 GT 的样本，也不会虚构六相机的共同标注。

可筛选相机，例如只比较前视图：

```bash
bash tasks/nuimages/scripts/segment_nuimages.sh \
  --cameras CAM_FRONT --max-images 5 \
  --output-dir outputs/nuimages_front
```

`--cameras` 支持多个六相机通道名称，默认不过滤。按图像 timestamp、sample token 排序；`--start-index` 和 `--max-images` 指向筛选后的 keyframe 列表。默认只处理 10 张，`--max-images 0` 才处理从起始索引到结尾的全部选定图像。输出目录须为新目录或空目录。

`--version v1.0-train/v1.0-val` 支持同格式数据，但需要自行提供对应 metadata 与图像；本地目前只有 mini。`v1.0-test` 没有可用真值，脚本只允许自定义类别模式进行推理。

### 类别与真值合成

使用 [nuimages_classes.txt](../tasks/nuimages/config/nuimages_classes.txt)，左侧保留原始类别名，右侧可改英文提示词并用 `|` 加别名。也可通过 `--classes-file` 指定完整的替代提示文件。保留官方 ID：

- ID 0：未标注背景；对象类别按字母排序为 1–23。
- ID 24：`flat.driveable_surface`；ID 31：`vehicle.ego`。
- ID 25–30：未使用的编号，保留占位但不提示、不评估。
- 默认不提示、不评估 ego，因此推理有 24 个有效类别；添加 `--include-ego` 后提示并评估 ID 31，共25类。

这些 ID **不能直接复用 nuScenes lidarseg 的类别映射**。每次运行的 `classes.json` 记录名称、ID、有效状态、提示词与颜色。

标注的 mask 是 base64 包装的 COCO compressed RLE，先 base64 解码再交给 pycocotools，尺寸须与图像一致。先按 JSON 顺序画 surface 掩码，再按 object annotation token 排序画对象掩码，使对象覆盖 surface，遵循官方 `get_segmentation()` 规则。null mask 跳过并统计，不根据检测框伪造掩码；未覆盖的像素为 GT0。

官方实现参考：[mask_decode / name_to_index_mapping](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuimages/utils/utils.py)、[get_segmentation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuimages/nuimages.py)。

### 评估范围与输出

默认忽略 GT0、GT31、自带未使用 ID25–30 和255。有效GT区域中预测0仍计为错误。按全部处理图像累计像素混淆矩阵，输出 mIoU、像素准确率、mean accuracy、frequency-weighted IoU、macro Dice、逐类 IoU/Dice/precision/recall/TP/FP/FN；宏平均与其它图像模式规则相同，指标值为0–1。

**nuImages 的标注不是完整场景语义类别全集**：图像中没有标注的背景区域忽略，不提供本实验所需的天空、建筑、植被等 GT 类别。指标只反映有效标注对象/表面像素，不能当作所有图像像素的场景分割效果。`metrics.json` 的 `evaluation_scope` 会记录总图像像素、已标注像素、有效GT占比、掩码计数、忽略策略和运行状态。本实现计算语义像素指标，不计算实例分割 AP。

```text
outputs/<nuimages_run>/
  predictions/<stem>.png         # 原始尺寸 uint8 类别ID预测
  ground_truth/<stem>.png        # 原始尺寸 uint8 官方合成语义GT
  instances/<stem>.json          # SAM3提示词/实例分数/类别/框等元数据
  visualizations/<stem>.jpg     # 3列×2行：原图、预测叠加、预测图、GT叠加、GT图、错误图
  overlays/<stem>.jpg           # 预测叠加
  color_masks/<stem>.png        # 预测彩色图
  gt_overlays/<stem>.jpg        # 真值叠加
  gt_color_masks/<stem>.png     # 真值彩色图
  errors/<stem>.png             # 正确绿、错误红、忽略/未标注灰
  classes.json / run_config.json / samples.jsonl
  metrics.json / metrics.csv / confusion_matrix.csv
```

`--no-vis` 不生成彩色图、叠加图、错误图和对比图，但仍保存预测、真值和评估；`--vis-every N` 降低可视化频率。中断/失败时保存累计指标并标记状态，查看 `run_config.json` 的 `status/processed_images`。

### TXT 自定义类别模式

```bash
bash tasks/nuimages/scripts/segment_nuimages_custom.sh \
  --classes-file shared/config/classes.txt --max-images 5 \
  --output-dir outputs/nuimages_custom
```

类别按 TXT 顺序分配1..N，0为unassigned，同其它自定义入口。该模式只读取图像元数据，不要求 annotation 表；输出预测、彩色图、叠加及3栏对比，不生成GT/错误图/语义评估，因为自定义类别未必对应标注。

```bash
# 检查选图、标注解码、类别ID及有效像素数，不加载SAM3
bash tasks/nuimages/scripts/segment_nuimages.sh --inspect-data

# 查看参数或运行nuImages回归测试
bash tasks/nuimages/scripts/segment_nuimages.sh --help
bash tasks/nuimages/scripts/test_nuimages.sh
```

## 6. 在 nuScenes mini 上运行 EFFOcc Fusion-R18

使用现有 **WSL `vla-supervised`** 环境加载 `assets/checkpoints/effocc_fusion_r18.pth`。Bash 入口默认只处理 **一个 LiDAR sample 及其六路相机**，输入为当前 LiDAR 关键帧加最多9个历史 sweeps。`--max-samples` 的单位是 sample，不是相机图像。适配入口无需修改数据集、生成 BEVDet pickle、下载模型或降级当前环境。

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# 只检查图像、标定、历史扫帧、点特征和深度图输入。
bash tasks/effocc/scripts/run_effocc_mini.sh --inspect-data

# 使用指定权重运行一个六相机样本。
bash tasks/effocc/scripts/run_effocc_mini.sh \
  --max-samples 1 \
  --output-dir outputs/effocc_demo

# 下一个样本有真实历史扫帧，而不是空历史补齐。
bash tasks/effocc/scripts/run_effocc_mini.sh \
  --start-index 1 --max-samples 1 \
  --output-dir outputs/effocc_history_demo

# 从mini全集选择场景。
bash tasks/effocc/scripts/run_effocc_mini.sh \
  --split all --scenes scene-0061 --max-samples 1 \
  --output-dir outputs/effocc_scene0061

# 查看参数并运行适配测试。
bash tasks/effocc/scripts/run_effocc_mini.sh --help
bash tasks/effocc/scripts/test_effocc.sh
```

默认读取 `datasets/nuscenes`，split 为 `mini_val`。用 `--dataset-dir`、`--checkpoint` 替换路径，每次使用新的或空的输出目录。`--max-samples 0` 处理从起始索引开始的全部选定样本。适配算子支持 `--device cpu` 做低速检查，真实样本已在CUDA验证。此入口使用float32，并在验证适配时关闭TF32；不接受文本类别，也不使用SAM3的 `--max-images/--precision` 参数。当前是 **相机 + LiDAR双模态融合模型**，同时使用图像特征、稀疏LiDAR特征和LiDAR投影深度。新推理默认将occupancy渲染到六路原始相机图像；添加 `--no-camera-projection` 可跳过渲染，`--projection-alpha 0.45` 调节叠加透明度。

### 权重和原生网格

给定权重与Occ3D配置 [flashocc_fusion_r18_base_100%_seqs.py](../models/EFFOcc/configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py) 严格匹配：387个state张量，没有缺失或多余参数。它是 **18类Occ3D**，不是17类OpenOccupancy配置。

- 六张输入图像缩放/裁剪为256x704。
- 融合前图像BEV特征为1x64x200x200，LiDAR BEV特征为1x256x200x200。
- 预测为uint8的 **200x200x16**，轴顺序X,Y,Z，输出体素大小0.4米。
- 原生范围X/Y为 `[-40,40)` 米，Z为 `[-1,5.4)` 米。17类为free，0..16保留原语义类别。
- LiDAR编码器的输入体素大小是0.05x0.05x0.16米，与0.4米输出占用体素不同。
- 二维BEV原生为 **200x200、80米x80米**。此前SAM3的500x500属于独立的点云栅格化设置，不能直接换进已训练网络。

```text
outputs/<effocc_run>/
  run_config.json               # 环境、checkpoint SHA256、严格加载状态和时间
  classes.json                  # 原始18类名称
  <scene_index_sample>/
    occupancy.npz               # semantics[X,Y,Z]、原生范围、体素大小和类别名称
    bev_class_ids.png           # 各XY列最高非free体素的类别ID，200x200
    bev_semantics.png           # 原生200x200语义彩色图，整列free为白色
    bev_preview.png             # 带类别图例和网格说明的二维放大图
    sample.json                 # 输入路径、扫帧选择、归一化、形状和有限logits检查
```

### 运行兼容性和验证范围

原始 [EFFOcc仓库](../models/EFFOcc/README.md) 依赖旧版MMCV 1.5.2..1.7.0、mmdet2/mmseg0 API、编译稀疏算子及自定义CUDA BEVPoolv2。当前Python3.12/torch2.14/cu132环境没有这些依赖，也没有nvcc，无法原样运行 `tools/test.py`。

因此本项目在 [effocc_model.py](../tasks/effocc/python/effocc_model.py)、[effocc_sparse.py](../tasks/effocc/python/effocc_sparse.py)、[effocc_data.py](../tasks/effocc/python/effocc_data.py) 增加了 **仅用于推理的PyTorch适配**。保留图像、稀疏LiDAR、深度、融合和占用预测分支，并严格加载全部权重。稀疏SubM/stride卷积通过活动坐标匹配和矩阵乘法计算，BEVPoolv2通过分块加权 `index_add_` 计算。原 `models/EFFOcc` 源码未修改；架构/流程保留EFFOcc、OpenMMLab、BEVDet来源说明和Apache-2.0许可背景。

依赖现有torch、torchvision、NumPy、Pillow和OpenCV（`cv2`）；[requirements.txt](../tasks/effocc/requirements.txt) 列出了另一环境需要的非torch补充项。本次没有安装或降级任何包。新环境应先配置匹配GPU的CUDA torch/torchvision；若与SAM3一起安装依赖，应遵守SAM3声明的NumPy约束，不要把补充文件当成环境锁定文件。

数据处理保留源码的实际行为，包括Pillow默认BICUBIC缩放/裁剪、上游 `mmlabNormalize(to_rgb=True)` 对PIL RGB反转为BGR、点云第5维为时间差而不是ring，以及点云参考CAM_FRONT与图像参考CAM_FRONT_LEFT的不同约定。适配器确定性选取最近9个历史sweeps，原配置可能随机选取；同像素深度采用稳定最小值，不复现float32 rank排序的量化缺陷。差异均记录在 `sample.json`。

**已验证：** 两个真实mini样本，包括首帧空历史补齐与下一帧的9个真实sweeps；387/387张量严格加载；完整分支logits有限；53项算子/输入/输出/渲染测试，包括CPU/CUDA稀疏卷积与稠密参考运算比较。这些检查证明推理能够运行。**尚未建立与原MMCV后端的整网输出等价性，也没有验证论文精度。**

本地lidarseg点标签不是Occ3D密集体素GT。预测和可视化不要求occupancy标签，但此命令有意 **不输出占用mIoU**。正式Occ3D评估需要匹配的密集 `labels.npz` 和有效性mask，不能将预览或稀疏lidarseg结果当作benchmark分数。

### 将occupancy投影到六路原始相机

新推理默认在各sample输出目录中创建 `camera_projections/`。也可以直接复用已有预测，不重新运行网络：

```bash
bash tasks/effocc/scripts/project_effocc_cameras.sh \
  --input-dir outputs/effocc_mini_verified \
  --max-samples 1 --alpha 0.45 \
  --output-dir outputs/effocc_camera_demo
```

`--input-dir` 可以是EFFOcc运行根目录，也可以是包含 `occupancy.npz` 和 `sample.json` 的单个样本目录。`--max-samples` 默认1，0渲染所有已保存样本；原数据根目录变化时用 `--dataset-dir` 指定。独立投影默认 `--device cuda`，也支持较慢的CPU。输出目录须为新目录或空目录。

```text
<sample-or-projection-output>/camera_projections/  # 自动推理输出
  CAM_FRONT_LEFT_original.jpg                    # 六路相机都使用相同命名规则
  CAM_FRONT_LEFT_overlay.jpg                     # 原始尺寸RGB图像加语义占用叠加
  CAM_FRONT_LEFT_semantics.png                   # 渲染表面颜色，无命中白色
  CAM_FRONT_LEFT_class_ids.png                   # uint8类别0..16，无命中255
  CAM_FRONT_LEFT_projection.npz                  # labels、camera-Z深度、voxel_index、camera_to_grid、intrinsic
  six_cameras_original.jpg                       # 3x2原图概览
  six_cameras_overlay.jpg                        # 3x2占用叠加概览及类别图例
  six_cameras_semantics.jpg                      # 3x2渲染语义表面
  projection_config.json                         # 网格参考系、相机token、变换、射线规则和状态
```

独立投影将这些文件放在 `<output-dir>/<sample-name>/`。单张图片保持原始相机尺寸（本地1600x900），拼图为方便查看而缩小。渲染对每个原图像素中心建立 `K^-1 [u+0.5,v+0.5,1]` 射线，通过精确体素遍历选取 **射线遇到的第一个占用体素**。17/free透明，0..16均为不透明表面（others0为灰色）；叠加图无命中像素保留原图。它投影实际体素体积并处理深度遮挡，不使用固定半径画点或图像膨胀。原生0.4米体素的阶梯边界仍可见。

默认网格参考系为 **CAM_FRONT_LEFT时间戳对应的ego坐标系**，与适配器 `sample.json` 记录的图像key-ego约定一致。各相机使用自身时间戳的ego pose和传感器标定：`camera_to_grid = inverse(key_ego_to_global) @ camera_ego_to_global @ camera_sensor_to_ego`。原occupancy轴为X,Y,Z，投影前不转为BEV显示轴。`--reference-camera` 可用于诊断时改参考时间戳，不改变模型。原LiDAR分支采用CAM_FRONT参考时间的差异，以及动态目标时间差仍是局限；这些图展示模型预测，不是图像分割真值或精度评估。

NPZ中的depth单位为相机Z方向的米（无命中0）；`voxel_index` 为 `(x * ny + y) * nz + z`（无命中-1）。渲染不会修改保存的occupancy。投影会增加耗时，只需要occupancy/BEV时可在推理命令加 `--no-camera-projection`。

### 加入SAM3分割并与occupancy投影对比

复用已保存的EFFOcc帧及对应六路原始相机图像，生成匹配的三列对比：**原图 / SAM3叠加 / EFFOcc occupancy叠加**。SAM3独立对完整原图分割，默认使用 [nuscenes_classes.txt](../tasks/nuscenes_sam3/config/nuscenes_classes.txt) 的16类提示词；共同语义ID 1..16使用一致配色，两种叠加透明度也相同。

```bash
bash tasks/effocc/scripts/compare_effocc_sam3.sh \
  --input-dir outputs/effocc_mini_with_cameras \
  --max-samples 1 --alpha 0.45 \
  --output-dir outputs/effocc_sam3_demo
```

输入可以是EFFOcc运行根目录或单个sample目录。检查sample和camera token后复用已有 `camera_projections/`；若投影保存于其它位置，传入 `--projection-dir outputs/effocc_six_camera_projection`。没有自动投影且未显式指定目录时，会在加载SAM3前先渲染缺失投影，不重新运行EFFOcc推理。默认 `--max-samples 1` 比较一个已保存LiDAR样本的六相机，0处理全部保存样本。

可选参数：`--sam3-checkpoint models/sam3/sam3.pt`、`--classes-file tasks/nuscenes_sam3/config/nuscenes_classes.txt`（保留完整nuScenes16类别）、`--score-threshold 0.5`、`--mask-threshold 0.5`、`--precision bf16`、`--device cuda`、`--dataset-dir`。CPU运行SAM3须同时指定 `--device cpu --precision fp32`，全部继续使用现有WSL `vla-supervised` 环境。

```text
<comparison-output>/
  run_config.json
  <sample-name>/
    classes.json / comparison_config.json
    CAM_FRONT_original.jpg
    CAM_FRONT_sam3_class_ids.png       # uint8 0..16，0为未分配
    CAM_FRONT_sam3_semantics.png       # 未分配区域白色
    CAM_FRONT_sam3_overlay.jpg         # 原尺寸独立SAM3分割
    CAM_FRONT_sam3_instances.json
    CAM_FRONT_occupancy_class_ids.png  # 未改变的投影ID 0..16/255
    CAM_FRONT_occupancy_semantics.png
    CAM_FRONT_occupancy_overlay.jpg
    CAM_FRONT_comparison.jpg           # 该相机同一帧的三栏对比
    six_cameras_sam3.jpg               # 3x2 SAM3概览
    six_cameras_comparison.jpg         # 六行三列，每行一个相机
```

六路相机均保存相同命名规则的文件。单相机掩码和叠加图保持原尺寸，对比图面板缩放用于查看。**SAM3的0是未分割到，EFFOcc的0是其它占用；17/free透明，投影255是射线无命中。** 不把这些标签合并。输入图像通过sample token和camera token精确匹配，不使用相邻帧替代。`comparison_config.json` 记录两者覆盖比例、运行配置中的阈值、提示词、图像路径和投影来源。这是两种预测的视觉比较，不是GT评估，也不把像素差异解释为某个模型的准确率。EFFOcc使用配置中的图像裁剪和有限占用范围，而SAM3使用完整原图。
