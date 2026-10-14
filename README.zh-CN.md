# vla-supervised-ad

[English](README.md) | **简体中文**

使用本地 SAM3 对 Sidewalk Semantic、nuScenes mini 和 nuImages 相机图像做文本提示分割，并导出类别图、可视化和评估指标。nuScenes 支持 lidarseg 投影评估和彩色点云，nuImages 使用图像掩码真值评估。

阅读顺序：[环境配置](#环境配置) → [可用命令总表](#可用命令总表) → [项目目录与数据路径](#项目目录与数据路径)。后续章节详细说明各数据集、点云输出和评估口径。

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

当前环境已具备全部运行依赖，日常运行可直接使用下面的 Bash 入口。[run_vla.sh](scripts/run_vla.sh) 会选择 `vla-supervised`，未手动激活时使用 `conda run -n vla-supervised`，不使用 base 或 Windows Python。解释器当前为 `/home/administrator/miniconda3/envs/vla-supervised/bin/python`。如果 Conda 装在其它位置，先激活该环境，或设置 `CONDA_EXE=/实际安装位置/bin/conda`。

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
| matplotlib | 3.11.2 | 点云静态预览 |
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
bash scripts/setup_sidewalk.sh
bash scripts/setup_nuimages.sh
```

[setup_sidewalk.sh](scripts/setup_sidewalk.sh) 根据 [requirements-sidewalk.txt](requirements-sidewalk.txt) 补充 PyArrow、Pillow、NumPy；[setup_nuimages.sh](scripts/setup_nuimages.sh) 根据 [requirements-nuimages.txt](requirements-nuimages.txt) 补充 pycocotools、Pillow、NumPy。这两个脚本不会安装 SAM3、torch 或 torchvision。nuScenes 数据读取与投影使用 NumPy/Pillow，不需要 `nuscenes-devkit`；点云 HTML 是离线 WebGL，无需 Web 服务或 CDN。静态点云 PNG 需要 matplotlib。

模型使用本地官方 `models/sam3/sam3.pt`。模型目录和 SAM3 源码目录分别是 `/mnt/d/model_zoo/sam3` 与 `/mnt/d/wsl_workplace/sam3`：权重目录不会自动提供 `sam3` Python 包。模型构建明确设置 `load_from_HF=False`，无需在运行分割时下载权重。

### 检查环境与数据

```bash
cd /mnt/d/wsl_workplace/vla-supervised-ad

# 输出实际解释器、GPU 可用性、关键库及 SAM3 源码位置；不加载模型。
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

# 检查链接。所有源路径须存在；已有链接不需要重新创建。
ls -ld models/sam3 datasets/sidewalk-semantic datasets/nuscenes datasets/nuimages
readlink -f models/sam3 datasets/sidewalk-semantic datasets/nuscenes datasets/nuimages

# 只读取数据/标注，不运行 SAM3。
bash scripts/segment_sidewalk.sh --inspect-data
bash scripts/segment_nuscenes.sh --inspect-data
bash scripts/segment_nuimages.sh --inspect-data
```

环境检查中 CUDA 不可用时，先在 WSL 中核对 `nvidia-smi` 和 torch 的 CUDA wheel。所有分割入口也支持 `--device cpu --precision fp32`，CPU 运行速度会慢于 GPU。

## 可用命令总表

以下命令均假定已进入项目根目录。每次分割或点云导出使用**新的或空的输出目录**。表中示例均限制样本数，不会启动全量推理；完整参数可对分割/导出入口追加 `--help`。

| 用途 | 可直接执行的命令 | 输出 / 默认行为 |
|---|---|---|
| Sidewalk 数据集类别分割与评估 | `bash scripts/segment_sidewalk.sh --max-images 5 --output-dir outputs/sidewalk_demo` | 预测、GT 对比可视化、像素指标；默认 train |
| Sidewalk TXT 自定义类别 | `bash scripts/segment_custom.sh --classes-file config/classes.txt --max-images 5 --output-dir outputs/sidewalk_custom` | 预测与可视化，不计算 GT 指标 |
| nuScenes 前视相机，16 类 | `bash scripts/segment_nuscenes.sh --max-images 5 --output-dir outputs/nuscenes_demo` | 默认 mini_val / CAM_FRONT；lidarseg 投影点观测评估 |
| nuScenes 原始 32 类 | `bash scripts/segment_nuscenes.sh --taxonomy lidarseg32 --max-images 5 --output-dir outputs/nuscenes_raw32` | 保留原始 lidarseg IDs |
| nuScenes 六相机 + 彩色点云 | `bash scripts/segment_nuscenes_6cams.sh --output-dir outputs/nuscenes_six_demo` | 默认 6 张相机图＝1 帧 LiDAR，自动融合、导出 3D 和唯一点指标 |
| nuScenes 六相机处理 10 帧 | `bash scripts/segment_nuscenes_6cams.sh --max-images 60 --output-dir outputs/nuscenes_six_10frames` | 60 张图像＝10 帧，各帧独立融合 |
| nuScenes TXT 自定义类别 | `bash scripts/segment_nuscenes_custom.sh --classes-file config/classes.txt --max-images 5 --output-dir outputs/nuscenes_custom` | 图像分割与可视化，不计算类别 GT 指标 |
| nuScenes 六相机 TXT 自定义类别 + 点云 | `bash scripts/segment_nuscenes_6cams.sh --mode custom --classes-file config/classes.txt --output-dir outputs/nuscenes_six_custom` | 预测点云与覆盖率，未计算 lidarseg 语义指标 |
| 已有 nuScenes 分割结果染色到点云 | `bash scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --output-dir outputs/nuscenes_demo_cloud` | 无需重跑 SAM3；PLY、NPZ、HTML、PNG 和数据集模式唯一点评估 |
| 仅导出已有结果中的 1 帧点云 | `bash scripts/color_nuscenes_pointcloud.sh --input-dir outputs/nuscenes_demo --max-frames 1 --output-dir outputs/nuscenes_first_cloud` | 数量单位为 LiDAR keyframe |
| nuImages 类别分割、GT 比较与评估 | `bash scripts/segment_nuimages.sh --max-images 5 --output-dir outputs/nuimages_demo` | 默认 v1.0-mini，24 个非 ego 类别；六栏对比、像素指标 |
| nuImages 仅筛选前视图 | `bash scripts/segment_nuimages.sh --cameras CAM_FRONT --max-images 5 --output-dir outputs/nuimages_front` | 只选已标注 key camera 图像 |
| nuImages 同时提示和评估 ego | `bash scripts/segment_nuimages.sh --include-ego --max-images 5 --output-dir outputs/nuimages_with_ego` | 包含 ID31，共25个提示类别 |
| nuImages TXT 自定义类别 | `bash scripts/segment_nuimages_custom.sh --classes-file config/classes.txt --max-images 5 --output-dir outputs/nuimages_custom` | 图像分割与可视化，不计算 GT 指标 |
| 建立/核对 nuImages 软链接 | `bash scripts/link_nuimages.sh` | 指向 `/mnt/d/datasets/nuimages`，已有不同目标时不覆盖 |
| 补充 Sidewalk 数据读取依赖 | `bash scripts/setup_sidewalk.sh` | 固定安装到 vla-supervised |
| 补充 nuImages 掩码解码依赖 | `bash scripts/setup_nuimages.sh` | 固定安装到 vla-supervised |
| 全部回归测试 | `bash scripts/test_sidewalk.sh` | 名称虽含 sidewalk，实际执行全部 tests；包含点云与 nuImages |
| nuScenes 数据/导出管线测试 | `bash scripts/test_nuscenes.sh` | 按 `*nuscenes*.py` 筛选，不单独包含 viewer/fusion 文件 |
| nuImages 数据/管线测试 | `bash scripts/test_nuimages.sh` | 按 `*nuimages*.py` 筛选 |
| 在固定环境执行 Python 命令 | `bash scripts/run_vla.sh -m pip list` | 通用环境入口，可传 Python 的任意参数 |

### 数量、参数与查看结果

| 参数 / 默认值 | Sidewalk | nuScenes 通用入口 | nuScenes 六相机入口 | nuImages |
|---|---|---|---|---|
| 默认处理数量 | 全部剩余图像 | 10 张相机图 | 6 张相机图＝1帧 | 10 张标注图 |
| `--max-images N` 单位 | 图像 | 相机图像 | 相机图像，建议6的倍数 | 筛选后的标注图 |
| `--max-images 0` | 全部剩余 | 全部选定视角 | 全部选定六视角 | 全部剩余 keyframes |
| 数据范围 | `--split train/validation/test` | `--split all/mini_train/mini_val` | 同 nuScenes | `--version v1.0-mini` / `v1.0-train` / `v1.0-val` / `v1.0-test` |
| 类别模式 | 数据集 / custom | `lidarseg16/lidarseg32` / custom | 同 nuScenes | 数据集 / custom；可含 ego |

三个数据集的分割入口都支持 `--start-index`、`--classes-file`、`--score-threshold`、`--mask-threshold`、`--device`、`--precision`、`--alpha`、`--vis-every`、`--no-vis`、`--inspect-data`、`--output-dir`。默认阈值均为0.5，默认 `cuda/bf16`。模型和数据路径可通过 `--checkpoint`、`--dataset-dir` 替换。不同 split/version 必须实际存在；本地 Sidewalk 只有 train，nuScenes 和 nuImages 只有 mini。

nuScenes 使用 `--cameras`、`--scenes`、`--min-depth`、`--export-pointcloud`；nuImages 的 `--cameras` 是筛选标注 key camera，不代表每个 sample 都有六张标注。点云独立导出入口支持 `--max-frames`、`--html-max-points`、`--no-html`、`--no-preview`；`--html-max-points` 只影响显示抽样，指标和 PLY/NPZ 保留全量。

- 二维对比图：各分割输出的 `visualizations/`；原始类别图位于 `predictions/`。
- 图像指标 / 相机投影点观测指标：分割输出根目录的 `metrics.json`、`metrics.csv`。
- 六相机点云：分割输出的 `pointcloud_results/index.html`，可直接在 Windows 浏览器中打开；点云 PLY/NPZ/PNG 位于其 `pointclouds/`。
- 融合后的唯一点指标：点云输出的 `evaluation/metrics.json`，分别报告 covered_points、full_scan 和 coverage，口径不同于分割输出根目录的投影观测指标。
- TXT 自定义模式不输出语义 GT 评估，nuScenes 自定义点云仍会报告几何覆盖率。

各模式的类别映射、输出字段和评估细节见后续章节。

## 项目目录与数据路径

仅支持 **WSL + Conda `vla-supervised`**。启动脚本会选择该环境，不使用 base 或 Windows Python。Python 实现放在 `python/`，`scripts/` 仅放 Shell 启动脚本，类别配置放在 `config/`。

```text
python/
  sam3_sidewalk.py          # 数据读取、SAM3 推理、实例合并、可视化
  sam3_nuscenes.py          # nuScenes 相机分割及 lidarseg 投影点评估
  sam3_nuimages.py          # nuImages 图像分割、GT 对比和像素指标
  nuimages_data.py          # nuImages keyframe 与 base64 RLE 掩码解码
  nuscenes_data.py          # mini 元数据、类别映射与跨时刻坐标投影
  nuscenes_pointcloud.py    # 相机像素类别回填、点云导出与唯一点评估
  pointcloud_semantics.py   # 跨相机融合、PLY、覆盖率和两种评估范围
  pointcloud_viewer.py      # 离线交互 3D HTML 与静态预览
  segmentation_metrics.py  # 混淆矩阵与语义分割评估
scripts/
  run_vla.sh               # 固定 WSL/vla-supervised 解释器
  setup_sidewalk.sh         # 补齐数据读取依赖
  segment_sidewalk.sh       # 数据集类别分割 + 评估
  segment_custom.sh         # 自定义类别分割
  segment_nuscenes.sh       # nuScenes lidarseg 类别 + 投影点评估
  segment_nuscenes_custom.sh # nuScenes 自定义类别分割
  segment_nuscenes_6cams.sh # 六视角分割、融合点云、三维结果与评估
  segment_nuimages.sh       # nuImages 类别分割、真值对比和评估
  segment_nuimages_custom.sh # nuImages 自定义 TXT 类别分割
  link_nuimages.sh          # 建立/核对 WSL 数据集软链接
  setup_nuimages.sh         # 在 vla-supervised 补充 RLE 依赖（当前已具备）
  test_nuimages.sh          # nuImages 数据读取和评估管线测试
  color_nuscenes_pointcloud.sh # 将已有像素类别回填点云、生成3D结果和评估
  test_sidewalk.sh          # 全部数据处理及评估回归测试
  test_nuscenes.sh          # nuScenes 投影和管线测试
config/
  sidewalk_classes.txt     # 原始数据集类别名 -> SAM3 英文提示词
  nuscenes_classes.txt     # nuScenes 官方 16 类 -> SAM3 提示词
  nuscenes_lidarseg32_classes.txt # 原始 32 类标签中的非零类别提示词
  nuimages_classes.txt     # nuImages 原始类别名 -> SAM3 提示词
  classes.txt              # 用户自定义类别示例
```

项目使用以下已有 WSL 符号链接：

- `models/sam3` → `/mnt/d/model_zoo/sam3`，使用官方 `sam3.pt`。
- `datasets/sidewalk-semantic` → `/mnt/d/datasets/sidewalk-semantic`，直接读取 `data/train-*.parquet` 和 `id2label.json`。
- `datasets/nuscenes` → `/mnt/d/datasets/nuscenes`，使用 `v1.0-mini/`、`samples/` 和 `lidarseg/v1.0-mini/`。
- `datasets/nuimages` → `/mnt/d/datasets/nuimages`，使用 `v1.0-mini/` 与 `samples/` 的二维标注图像。

当前环境的激活、依赖补充和新建环境步骤见前面的[环境配置](#环境配置)。下面按数据集介绍类别配置、输出和评估。

## 1. 使用 Sidewalk Semantic 类别并评估

先用少量图像确认输出：

```bash
bash scripts/segment_sidewalk.sh \
  --max-images 5 \
  --output-dir outputs/sidewalk_demo
```

评估全部 1000 张训练集图像：

```bash
bash scripts/segment_sidewalk.sh \
  --output-dir outputs/sidewalk_full
```

默认读取原始 34 个非零类别，保留数据集 ID 1–34；ID 0 为 `unlabeled`。`config/sidewalk_classes.txt` 将 `flat-sidewalk` 等数据集标签名映射到 `sidewalk` 等更适合 SAM3 的英文提示词。可以修改 `=` 右侧的提示词、用 `|` 添加别名；左侧必须保留完整的原始类别名。也可以使用 `--classes-file config/your_sidewalk_prompts.txt` 指定完整的替代映射。

原始类别包含 `void-ground`、`void-dynamic`、`void-static`、`void-unclear`。这些标签缺乏精确的自然语言视觉概念，配置中使用近似提示词，但评估仍按原始 ID 计分。这是 SAM3 文本提示的零样本语义分割评估，没有训练语义分类头；提示词、置信度阈值和掩码冲突处理均会影响指标。

本地数据仅有 `train` 划分。脚本也接受 `--split validation/test`，需要用户另行提供同格式的对应 Parquet 文件。按子集运行时，指标只覆盖实际处理的图像，不代表整个数据集。

## 2. 使用 TXT 自定义类别

编辑 `config/classes.txt`，每行一个英文类别：

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
bash scripts/segment_custom.sh \
  --classes-file config/classes.txt \
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
bash scripts/segment_sidewalk.sh --inspect-data

# 从全局索引 100 开始处理 10 张，降低输出可视化数量
bash scripts/segment_sidewalk.sh \
  --start-index 100 --max-images 10 --vis-every 5 \
  --score-threshold 0.5 --mask-threshold 0.5 \
  --output-dir outputs/sidewalk_subset

# 查看完整参数
bash scripts/segment_sidewalk.sh --help

# 回归测试：类别配置、掩码合并、Parquet 读取、可视化、评估指标
bash scripts/test_sidewalk.sh
```

默认 `--device cuda --precision bf16`，适合本机 RTX 5070 Ti。支持 `--precision fp32/fp16`；CPU 模式需同时指定 `--device cpu --precision fp32`。`--no-vis` 只保存类别 ID 图、实例信息和评估结果。`--max-images 0` 表示处理从起始索引到结尾的全部图像。

## 3. nuScenes mini：相机分割与 lidarseg 投影点评估

仍然仅在 WSL 的 `vla-supervised` 环境运行，复用本地 SAM3。读取 JSON 与点云投影使用 NumPy/Pillow，不需要安装 `nuscenes-devkit` 或修改已有环境。

默认使用 `mini_val` 的 `CAM_FRONT`，**最多处理 10 张图像**，类别使用官方 32→16 类合并。你可以先限制为 1–5 张：

```bash
bash scripts/segment_nuscenes.sh \
  --max-images 5 \
  --output-dir outputs/nuscenes_demo
```

选择 mini 全集中的一个场景和多个相机：

```bash
bash scripts/segment_nuscenes.sh \
  --split all --scenes scene-0061 \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
  --max-images 6 \
  --output-dir outputs/nuscenes_scene0061
```

`--split` 支持 `mini_train`、`mini_val`、`all`。场景筛选与 split 取交集。六相机可选 `CAM_FRONT`、`CAM_FRONT_LEFT`、`CAM_FRONT_RIGHT`、`CAM_BACK`、`CAM_BACK_LEFT`、`CAM_BACK_RIGHT`。只使用 keyframe，与同一个 sample 的 `LIDAR_TOP` 和 lidarseg 标签关联，不使用 sweeps。按场景名、sample 时间戳、指定相机顺序枚举。

本地 mini 共 404 个 sample：全六相机为 2,424 张图像；mini_train 为 323 个 sample，mini_val 为 81 个 sample。**`--max-images` 和 `--start-index` 的单位是选定列表中的相机图像**，例如六相机配置下 `--max-images 6` 处理一个 sample 的六个视角。`--max-images 0` 才处理从起始索引到结尾的全部选定视角。每次运行需使用新的或空的输出目录。

### 类别模式

- 默认 `--taxonomy lidarseg16`：16 个有效类别，ID 1–16；ID 0 为 ignore/未覆盖。使用 `config/nuscenes_classes.txt`，保留左侧类别名，右侧可修改提示词或添加 `|` 别名。类别为 barrier、bicycle、bus、car、construction_vehicle、motorcycle、pedestrian、traffic_cone、trailer、truck、driveable_surface、other_flat、sidewalk、terrain、manmade、vegetation。
- `--taxonomy lidarseg32`：保留原始标签 ID 0–31，0=noise 被忽略，1–31 全部提示并评估。使用 `config/nuscenes_lidarseg32_classes.txt`。人群细分类及 `vehicle.ego` 等文本提示存在语义歧义；此模式是原始类别实验，官方 benchmark 通常使用合并后的 16 类。
- `--classes-file` 可以替换该 taxonomy 的完整提示配置；类别名与 IDs 不改变。

```bash
# 原始 lidarseg 32 类
bash scripts/segment_nuscenes.sh \
  --taxonomy lidarseg32 --max-images 5 \
  --output-dir outputs/nuscenes_raw32

# 自定义 TXT 类别，仅分割和可视化，不计算 lidarseg 指标
bash scripts/segment_nuscenes_custom.sh \
  --classes-file config/classes.txt --max-images 5 \
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
bash scripts/segment_nuscenes.sh --inspect-data

# 查看全部参数
bash scripts/segment_nuscenes.sh --help

# 投影、类别、稀疏可视化与指标测试
bash scripts/test_nuscenes.sh
```

## 4. 把像素类别染色到原始 LiDAR 点云

像素预测通过同一套标定投影回填到**原始点索引**，每个 LiDAR 点保存预测类别、GT、覆盖状态和颜色；坐标保持原始 `LIDAR_TOP` 传感器坐标，单位米。未投影到已处理相机的点为灰色，已覆盖但 SAM3 没有给出类别的点为浅灰色，其它点按像素类别配色，与二维图的类别颜色一致。

### 已有分割结果直接导出，无需重新推理

```bash
bash scripts/color_nuscenes_pointcloud.sh \
  --input-dir outputs/nuscenes_demo \
  --output-dir outputs/nuscenes_demo_cloud
```

输入为已完成的 nuScenes 分割输出目录，脚本读取 `samples.jsonl`、`predictions/` 和配置。也可以从失败/中断的运行导出已记录完成的图像，来源状态会保留在输出中；不要在源分割任务仍运行时导出。自定义类别模式也可导出预测点云和覆盖率，但不计算与 lidarseg 的类别指标。

`--max-frames 1` 只导出一个 LiDAR keyframe；默认 0 导出输入中全部已完成的点云帧。`--dataset-dir` 可以替换源配置中的数据根目录。使用新的或空的输出目录，避免混入不同推理结果。

### 一次运行六相机分割、染色和评估

六视角专用入口默认处理一个 keyframe（6 张相机图），自动导出点云：

```bash
bash scripts/segment_nuscenes_6cams.sh --output-dir outputs/nuscenes_six_demo
```

若处理 10 帧，指定 `--max-images 60`；建议 `--max-images` 和 `--start-index` 使用 6 的倍数，避免首尾帧只有部分视角。也可使用通用入口：

```bash
bash scripts/segment_nuscenes.sh \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \
  --max-images 6 \
  --export-pointcloud \
  --output-dir outputs/nuscenes_one_scan
```

这里 6 张图像对应一个 sample 的六个相机，最终融合成一帧彩色点云。默认 `--split mini_val --taxonomy lidarseg16`，可使用原始 `lidarseg32` 或自定义类别入口。`--export-pointcloud` 接在原来的分割命令上即可，自动导出到 `<output-dir>/pointcloud_results/`。已有单相机结果只包含该相机覆盖的区域；点云导出不会自动运行其它相机。相机列表被 `--start-index`/`--max-images` 截断时，该帧仅融合实际完成的视角，输出中会记录相机名单。

同一个原始点在多个相机中出现时，采用**非零类别多数票**；0 表示未分割到，作为弃票，但点仍标记为几何覆盖。平票时优先输入清单中较早的相机。每个点在融合后只评估一次，避免重复计算相机重叠区域。融合不读取 GT，不用真值挑选预测。

### 查看和使用点云文件

```text
<pointcloud-output>/
  index.html                  # 每帧离线3D视图入口
  run_config.json             # 来源、状态、已处理点云帧数
  frames.jsonl                # 每帧相机名单与覆盖统计
  evaluation/
    metrics.json / metrics.csv # 唯一点：covered_points 和 full_scan 两套指标
    coverage.csv              # 逐类别GT覆盖率
    confusion_matrix_covered.csv
    confusion_matrix_full_scan.csv
  pointclouds/<scene_frame_sample>/
    viewer.html               # 交互旋转/缩放/平移；预测/GT/错误/覆盖范围切换
    preview.png               # 俯视和斜视的静态对比图
    prediction.ply            # 全部点，按预测类别染色，未覆盖点灰色
    prediction_covered.ply    # 仅相机覆盖点，含prediction0点
    ground_truth.ply          # 整帧GT类别染色，仅数据集类别模式
    errors.ply                # covered且GT非零点：正确绿、错误红；其它点深灰
    points.npz                # 完整原始点序列、类别、覆盖、融合来源等数组
    summary.json              # 当前帧统计、覆盖率和评估
    evaluation/               # 当前帧两套指标，仅数据集类别模式
```

`viewer.html` 和 `index.html` 自包含，可在浏览器直接打开，不需要 Web 服务或联网；浏览器需支持 WebGL。左键拖动旋转、右键/Shift 拖动平移、滚轮缩放，支持只看覆盖点和修改点大小。HTML 默认最多绘制 60,000 个点，超出时按原始点序列确定性均匀抽样；界面标明实际绘制数量。可用 `--html-max-points` 修改。**PLY、NPZ、PNG 和评估均保留全量点**，不因 HTML 显示抽样而改变指标；全帧不裁点。

PLY 可用 CloudCompare、MeshLab 或其它 PLY 阅读器打开。保留 `class_id`、`covered`、`source_camera` 属性；source_camera 对应 `points.npz` 中 camera_names 的索引，预测0时为 -1。NPZ 包含 `xyz/intensity/ring/raw_labels/point_indices/prediction/covered/observation_count/winning_votes/source_camera/camera_names`，数据集类别模式还包含 `target`；数组与原始 LiDAR 文件点顺序一致。

### 点云评估口径

- `covered_points`：仅评估至少一个已处理相机有几何投影覆盖的唯一点，忽略 GT0。已覆盖但预测0仍为错误，不能通过只选成功染色点抬高指标。
- `full_scan`：整帧所有非零 GT 点都计入，未覆盖点预测0作为错误。它反映实际填充整帧的能力；单相机时该分数通常明显较低。
- `coverage`：报告总点数、覆盖点数、实际有非零预测点数、有效 GT 覆盖率及逐类别覆盖率。每帧和全部已导出帧都有指标；GT 合并策略仍由源配置中的 16/32 类模式决定。
- 两种范围都输出 point accuracy、mIoU、mean accuracy、frequency-weighted IoU、macro Dice 和逐类 precision/recall/IoU/TP/FP/FN。0预测造成对应GT的FN；GT0完全排除宏指标和混淆矩阵。多相机同一点只计一次，每帧点索引独立。
- 标定、跨像素遮挡和动态物体时间差仍有误差。点云结果来自相机预测，不是训练的3D语义分割网络输出，也不是官方 nuScenes benchmark 提交分数。

`--no-html` / `--no-preview` 可分别关闭交互文件和静态图；分割命令中 `--export-pointcloud --no-vis` 会导出 PLY/NPZ 与评估，但关闭 PNG/HTML。点云指标位于点云输出的 `evaluation/`，原分割输出根目录的 `metrics.json` 仍按每相机投影点观测计数，二者口径不同。

## 5. nuImages：图像分割、真值对比、评估与可视化

本地数据为 `D:\datasets\nuimages` 的 **v1.0-mini，50 张已标注 keyframe**。已创建 WSL 软链接 `datasets/nuimages -> /mnt/d/datasets/nuimages`；可重复执行下面的命令核对/建立链接，已有不同链接或目录时不会覆盖：

```bash
bash scripts/link_nuimages.sh
```

当前 WSL `vla-supervised` 已具备 SAM3、PyTorch、Pillow、NumPy 和 RLE 解码用的 `pycocotools`，无需另装 devkit。另一个同名环境若缺少解码依赖，可执行 `bash scripts/setup_nuimages.sh`，补充项记录在 `requirements-nuimages.txt`。

### 数据集类别模式

```bash
bash scripts/segment_nuimages.sh \
  --max-images 5 \
  --output-dir outputs/nuimages_demo
```

默认 `--version v1.0-mini`、最多 10 张图，选择全部相机通道中的**已标注 key camera**。本地 mini 的 keyframe 分布为前8、前左5、前右7、后11、后左10、后右9。nuImages 每个 sample 提供一张标注 key camera，以及该相机前后的未标注 sweeps；这与 nuScenes 一个 sample 的六个同步相机不同。脚本不会把 sweeps 当作有 GT 的样本，也不会虚构六相机的共同标注。

可筛选相机，例如只比较前视图：

```bash
bash scripts/segment_nuimages.sh \
  --cameras CAM_FRONT --max-images 5 \
  --output-dir outputs/nuimages_front
```

`--cameras` 支持多个六相机通道名称，默认不过滤。按图像 timestamp、sample token 排序；`--start-index` 和 `--max-images` 指向筛选后的 keyframe 列表。默认只处理 10 张，`--max-images 0` 才处理从起始索引到结尾的全部选定图像。输出目录须为新目录或空目录。

`--version v1.0-train/v1.0-val` 支持同格式数据，但需要自行提供对应 metadata 与图像；本地目前只有 mini。`v1.0-test` 没有可用真值，脚本只允许自定义类别模式进行推理。

### 类别与真值合成

使用 `config/nuimages_classes.txt`，左侧保留原始类别名，右侧可改英文提示词并用 `|` 加别名。也可通过 `--classes-file` 指定完整的替代提示文件。保留官方 ID：

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
bash scripts/segment_nuimages_custom.sh \
  --classes-file config/classes.txt --max-images 5 \
  --output-dir outputs/nuimages_custom
```

类别按 TXT 顺序分配1..N，0为unassigned，同其它自定义入口。该模式只读取图像元数据，不要求 annotation 表；输出预测、彩色图、叠加及3栏对比，不生成GT/错误图/语义评估，因为自定义类别未必对应标注。

```bash
# 检查选图、标注解码、类别ID及有效像素数，不加载SAM3
bash scripts/segment_nuimages.sh --inspect-data

# 查看参数或运行nuImages回归测试
bash scripts/segment_nuimages.sh --help
bash scripts/test_nuimages.sh
```
