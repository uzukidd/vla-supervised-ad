# vla-supervised-ad

[English](README.md) | **简体中文**

使用本地 SAM3 和 EFFOcc 完成图像语义分割、点云语义 BEV、占用预测及平地逆投影。项目划分为 **五类任务**；每类下方提供一键命令和真实运行的最终效果图。

所有命令从项目根目录的 **WSL Bash** 执行，入口自动使用已有 Conda **`vla-supervised`** 环境及本地权重。默认处理一张图或一帧六相机，不运行全数据集。

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad
```

| 任务 | 输入与目标 | 一键入口 |
|---|---|---|
| [1. Sidewalk Semantic](#task-sidewalk) | 图像分割，与像素 GT 比较 | `bash tasks/sidewalk/scripts/run.sh` |
| [2. nuImages](#task-nuimages) | 已标注相机图像分割与评估 | `bash tasks/nuimages/scripts/run.sh` |
| [3. nuScenes SAM3 与 BEV](#task-nuscenes) | 六相机分割、LiDAR 语义融合、道路/人行道铺面 | `bash tasks/nuscenes_sam3/scripts/run.sh` |
| [4. EFFOcc 与 SAM3 对比](#task-effocc) | 图像 + LiDAR 占用预测，投影到同帧相机比较 | `bash tasks/effocc/scripts/run.sh` |
| [5. 平地语义逆投影](#task-flat-ground) | 假设全局 z=0，把相机语义块直接投到地面 | `bash tasks/flat_ground/scripts/run.sh` |

详细安装、完整参数、数据路径和评估口径见 [中文详细指南](docs/guide.zh-CN.md)；[English guide](docs/guide.en.md)。以下图片均为实际输出的原图副本，保存在 [文档图片目录](docs/images/)，不依赖被忽略的实验输出目录。

<a id="task-sidewalk"></a>
## 1. Sidewalk Semantic：图像分割与像素评估

使用 SAM3 按 Sidewalk 数据集类别分割图像，输出语义类别图、叠加图、与数据集 GT 的对比，以及 mIoU、逐类 IoU 等像素指标。默认运行一张图。

```bash
bash tasks/sidewalk/scripts/run.sh
```

**最终效果：** 同一张图的原图、SAM3 预测与数据集真值对比。

![Sidewalk Semantic 最终效果：原图、SAM3 分割与 GT](docs/images/sidewalk.jpg)

类别提示词：[sidewalk_classes.txt](tasks/sidewalk/config/sidewalk_classes.txt)。输出位于 `outputs/sidewalk/<时间戳>/`，打开 `result.html` 查看，最终图片在 `results/`。高级自定义类别与评估规则见 [详细指南](docs/guide.zh-CN.md#1-使用-sidewalk-semantic-类别并评估)。

<a id="task-nuimages"></a>
## 2. nuImages：相机图像分割与真值对比

对 nuImages 有标注的相机关键帧做 SAM3 分割，解码数据集掩码，输出预测、GT 与错误图，并在有效标注像素上评估。默认运行一张图；不同样本不要求六视角均有标注。

```bash
bash tasks/nuimages/scripts/run.sh
```

**最终效果：** 上排为原图、SAM3 叠加和类别图；下排为 GT 叠加、GT 类别图和预测错误图。

![nuImages 最终效果：SAM3、数据集真值及错误图](docs/images/nuimages.jpg)

类别提示词：[nuimages_classes.txt](tasks/nuimages/config/nuimages_classes.txt)。输出位于 `outputs/nuimages/<时间戳>/`，查看 `result.html` 和 `results/`。原始类别、ego 与忽略区域处理见 [详细指南](docs/guide.zh-CN.md#5-nuimages图像分割真值对比评估与可视化)。

<a id="task-nuscenes"></a>
## 3. nuScenes SAM3：六相机分割、语义 BEV 与地面铺面

六相机 SAM3 分割后，通过标定和位姿把类别回填至原始 LiDAR 点，跨相机融合，再生成语义 BEV。道路、人行道可在有点支持的局部三角网中铺成连续平面；补面来源单独保存。

```bash
# 标准 16 类：六相机、稀疏 lidarseg 评估、语义 BEV 与地面平面。
bash tasks/nuscenes_sam3/scripts/run.sh
```

如果只需要最近约定的 **sidewalk、road、pedestrian、vehicle、bike、rider 六类**，使用同一任务下的六类模式；只有 sidewalk、road 铺面，其余类保留观测语义：

```bash
bash tasks/nuscenes_sam3/scripts/run_six_classes.sh
```

**最终效果（六类模式示例）：** 左侧为点云语义 BEV，中间为 road/sidewalk 铺面后的六类 BEV，右侧标记观测、推断填充及非地面证据。标准 16 类模式还会生成独立 GT BEV 和对比图。

![nuScenes 最终效果：语义 BEV、道路人行道铺面与来源图](docs/images/nuscenes.png)

默认范围 **100 m × 100 m**，粒度 **0.2 m**，对应 **500 × 500** 栅格；原生 LiDAR 坐标 +x 向右、+y 向上。标准入口输出至 `outputs/nuscenes_sam3/<时间戳>/`，`result.html` 顶部是 BEV 分区，图片集中到 `results/BEV/<帧>/`。六类入口输出至 `outputs/nuscenes_six_classes/<时间戳>/`，结果页优先展示 BEV。

提示词：[16 类配置](tasks/nuscenes_sam3/config/nuscenes_classes.txt)、[六类配置](tasks/nuscenes_sam3/config/six_classes.txt)。vehicle 包含汽车/卡车/公交/挂车，bike 包含自行车/摩托车，rider 是骑乘者本人。六类模式不直接沿用 16 类指标，lidarseg 也没有独立 rider 真值。

同属本任务的扩展实验：

| 扩展 | 入口 | 用途 |
|---|---|---|
| 实例点凸包 | `bash tasks/nuscenes_sam3/scripts/run_instances.sh` | 保留相机实例，把分配点覆盖成凸包，含独立合并/地面结果 |
| 已有结果跨相机合并 | `bash tasks/nuscenes_sam3/scripts/merge_instances.sh --input-dir <已有实例输出>` | 同类前景共享点关联、点集去重、重算凸包 |
| 已有结果补地面平面 | `bash tasks/nuscenes_sam3/scripts/run_ground.sh --input-dir <已有输出>` | 复用分割/点预测，不重跑 SAM3 |
| 刷新旧一键结果页 | `bash tasks/nuscenes_sam3/scripts/run.sh --refresh-results --output-dir <已有一键输出>` | 补地面图并集中展示 BEV |

平面填充不是新增观测；稀疏投影点评估不是密集图像 GT。完整融合、边界与指标说明见 [详细指南](docs/guide.zh-CN.md)。

<a id="task-effocc"></a>
## 4. EFFOcc：占用预测与 SAM3 同帧对比

EFFOcc Fusion-R18 同时使用六相机图像和 LiDAR/历史扫帧预测三维语义占用，再把占用射线投影回六相机，与独立 SAM3 分割比较。一条命令自动完成两个模型阶段和对比图。

```bash
bash tasks/effocc/scripts/run.sh
```

**最终效果（前视相机示例）：** 左为原图，中为 SAM3 图像分割，右为同帧 EFFOcc 占用投影。运行结果还包含完整六相机对比图。

![EFFOcc 最终效果：原图、SAM3 与同帧占用投影](docs/images/effocc.jpg)

输出位于 `outputs/effocc/<时间戳>/`：`result.html` 展示最终对比，图片在 `results/`，原始占用与投影在 `raw/`，SAM3 对比在 `comparison/`。SAM3 提示词使用 [nuScenes 16 类配置](tasks/nuscenes_sam3/config/nuscenes_classes.txt)；EFFOcc 本身没有文本提示词，类别由训练权重固定。

当前是严格加载检查点的推理用 PyTorch 适配实现，没有完成与原始 MMCV/spconv 后端的全量数值等价验证。本地缺少密集占用 GT；两模型差异只作视觉比较，不是准确率。详见 [EFFOcc 指南](docs/guide.zh-CN.md#6-在-nuscenes-mini-上运行-effocc-fusion-r18)。

<a id="task-flat-ground"></a>
## 5. 平地语义逆投影：直接投到 z=0 地面

假设地面为 **nuScenes 全局地图坐标 z=0** 的水平面，用相机内外参及各自时刻的位姿，把 SAM3 语义块直接映射到地面。采用 BEV 格子中心反查相机像素的 IPM，不读取 LiDAR 点、不做三角网补面。默认只投 road、sidewalk。

```bash
bash tasks/flat_ground/scripts/run.sh
```

**最终效果：** 左为融合后的平地语义 BEV，中为几何相机覆盖，右为最终标签的来源相机。该图与任务 3 的点云 BEV 坐标不同：+X 是自车前方（图中向右），+Y 是左侧（图中向上）。

![平地投影最终效果：语义平面、相机覆盖与来源](docs/images/flat_ground.png)

默认仍为 **100 m × 100 m、0.2 m、500 × 500**，输出至 `outputs/flat_ground/<时间戳>/`，打开 `result.html`。可复用任务 3 保存的六类掩码，无需重新推理：

```bash
bash tasks/flat_ground/scripts/run.sh \
  --input-dir outputs/nuscenes_six_class_planes \
  --plane-z 0 \
  --output-dir outputs/flat_ground_demo
```

`--plane-z` 指全局地图高度，不是海平面绝对海拔。坡度、路沿高度、遮挡和分割误差会造成变形；白色区域表示缺少有效地面标签。若用 `--project-classes sidewalk road pedestrian vehicle bike rider` 投全部六类，高于地面的物体会被拉伸，不能视为真实接地轮廓。投影公式、参数及证据文件见 [平地 IPM 指南](docs/guide.zh-CN.md#平地ipm把语义块直接投到-z0)。

## 环境、参数与详细说明

- 环境固定使用 WSL 的 `vla-supervised`；[环境配置](docs/guide.zh-CN.md#环境配置) 中保留完整安装、依赖和数据检查步骤。
- 五个任务均默认只跑一个样本。使用 `--samples N` 扩大范围，`--start-index I` 选择起始位置，`--output-dir` 指向新/空目录；每个入口的 `--help` 显示实际支持的参数。
- Sidewalk、nuImages、标准 nuScenes 和 EFFOcc 一键入口的最终图集中于 `results/`；六类与平地任务直接按帧保存最终图，并提供同样可打开的 `result.html`。
- 运行全项目验证：`bash shared/scripts/test_all.sh`。环境、权重、数据和历史实验输出均不随文档调整而移动。

[完整命令参考](docs/guide.zh-CN.md#可用命令总表) · [项目目录与数据路径](docs/guide.zh-CN.md#项目目录与数据路径) · [English detailed guide](docs/guide.en.md)
