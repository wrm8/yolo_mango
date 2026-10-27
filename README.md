# 芒果颜色识别与单个小果选择

代码支持黄色、绿色两类芒果实例分割。**当前没有芒果数据集和训练权重，不能直接声称已经能准确识别实物。** 补齐数据后，按下面命令训练、评估并运行。

## 最终大小规则

正面完整画面必须检测到四个芒果，树干左边两个、右边两个。

- **只看检测框高度 `y2 - y1`，不看面积，也不比较果实在画面中的上下位置。**
- 左右分别比较同侧两个果实。高度相同的一侧跳过；高度不同的一侧选较矮者。
- 最终只有 **0 或 1 个小果**。两侧都不同时返回不确定，不强行从两侧候选中挑一个。
- 颜色独立输出。默认只将小果放进 `targets`，不会因为果实是绿色就选中它。

| 左侧框高 | 右侧框高 | 结果 |
|---|---|---|
| 100、70 | 120、120 | 左侧高度 70 的是小果 |
| 50、50 | 120、90 | 右侧高度 90 的是小果；不跨侧比大小 |
| 100、100 | 120、120 | 没有小果，confirmed，目标为空 |
| 100、70 | 120、90 | 无法确定唯一小果，uncertain，目标为空 |

漏检、多检、左右不是各两个、中心线附近有果、非法数值或置信度不足，同样返回不确定。

配置在 `configs/vision.json`：

- `selection_mode: "per_side"`：上面的当前规则。
- `height_mode: "pixel"`：默认比较像素框高。
- `center_x_ratio: 0.5`：左右分界的位置；应对准实际树干。
- `center_deadband_ratio: 0.02`：分界两侧各 2% 宽度属于不确定区。
- `tie_relative_tolerance: 0.0`：默认相等才跳过（仅忽略浮点舍入误差）。
  实测相同实物的框高通常会抖动，可按拍摄数据调整；例如 0.03 表示
  `1 - 较矮高度 / 较高高度 <= 0.03` 就按相同处理。0.03 只是示例，尚未标定。
- `confidence: 0.6`：最低检测置信度。
- `stable_frames: 8`：实时观测连续一致 8 帧后设置稳定标记。

框高会受距离和姿态影响。默认规则衡量图像中的高度，不等同于真实物理尺寸。
可选 `depth_scaled` 比较“框高 × 深度”，仍然是高度特征；缺深度时返回不确定。
普通照片、视频和 USB 彩色相机默认使用 pixel。

保留旧接口供已有程序使用：`unique_small` 使用 `small_ratio/similar_ratio` 的严格阈值规则，
要求恰好一个小果；这些阈值**不参与默认 per_side 规则**。
`target_policy: "green_or_small"` 可恢复旧的“绿色果和小果取并集”目标策略；
它可能输出多个绿色目标，但小果仍最多一个。当前默认是 `small_only`。

## 文件入口

| 文件 | 用途 |
|---|---|
| setup_wsl.sh | 安装训练环境；备份不可用的旧虚拟环境 |
| check_dataset.py、dataset_config.py | 多边形标签、颜色类别、图片解码检查及路径解析 |
| train.py | YOLO11n 实例分割训练 |
| evaluate.py | 验证集颜色检测/分割指标及分类别指标 |
| decision.py | 左右框高比较，最多选一个小果 |
| stability.py | 连续帧颜色、目标和位置一致性检查 |
| vision.py、rgb_source.py | 图片、目录、视频、USB 相机、RealSense 推理 |
| demo.py | 无权重、无硬件的合成检测结果演示 |
| run.sh | 检查、训练、评估、推理、测试的统一入口 |
| tests/ | 规则、数据和图像/视频流程测试 |

## 1. 安装环境

在 WSL Ubuntu 中执行：

```bash
cd /home/wrm/yolo_mango
bash setup_wsl.sh
source .venv/bin/activate
```

需要 Python 3.10+、curl 和网络。安装脚本不会使用复制来的 Python 3.12 环境；
若旧环境无法运行，会改名备份后重建。默认安装 PyTorch 2.7.1 / torchvision 0.22.1 的 CUDA 11.8 包。
只使用 CPU 时：

```bash
MANGO_TORCH_INDEX=https://download.pytorch.org/whl/cpu bash setup_wsl.sh
```

GPU 可用性以 `python check_environment.py --require-cuda` 的现场输出为准。
`requirements-wsl.lock.txt` 和原 `outputs/environment.json` 来自旧环境，不代表当前电脑已验证成功。
详情见 [TRAINING.md](TRAINING.md)。

## 2. 后续补充数据

目录已经预留：

```text
datasets/mango/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

类别固定：`0 = yellow_mango`（黄色），`1 = green_mango`（绿色）。
所有小果仍然按颜色标注，**不需要另设大小类别**。
每个芒果沿轮廓标注一个多边形，导出 YOLO 实例分割格式：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

每个实例一行，坐标归一化到 0～1，至少三个顶点；
五列检测框标签不适用于此分割模型。图片与标签同名，负样本用同名空标签文件。
两个颜色都应出现在训练和验证集中，按拍摄批次划分，避免相邻视频帧泄漏。
训练照片可以不是四果；“恰好四果”是运行时的判断前提。
标签检查可检查结构、零面积和图片能否解码，不能判断语义标注是否正确。

## 3. 检查、训练、评估

```bash
bash run.sh check
bash run.sh train --epochs 100 --imgsz 640 --batch 4
# 自动使用可用 GPU，也可指定 --device cpu 或 --device 0
bash run.sh evaluate --weights runs/segment/mango/weights/best.pt --device cpu
```

默认预训练起点 `yolo11n-seg.pt` 首次可能需要下载。
也可用 `--model /path/to/yolo11n-seg.pt` 指定本地权重。
输出目录重复时可能递增，以训练日志里的实际 `best.pt` 路径为准。
评估输出位于 `runs/segment/evaluation*/`，包含图表和 `evaluation.json`。
这些是颜色检测/分割指标，小果规则还需要用四果实拍场景单独核对。

## 4. 图片、视频、相机

```bash
# 单张图片
bash run.sh detect --weights runs/segment/mango/weights/best.pt --source test.jpg

# 图片目录，每张分别输出标注图和 JSON
bash run.sh detect --weights runs/segment/mango/weights/best.pt --source test_images --output outputs/batch

# 视频，保存逐帧记录和标注视频
bash run.sh detect --weights runs/segment/mango/weights/best.pt --source test.mp4 --save-video

# 普通 USB 摄像头
bash run.sh detect --weights runs/segment/mango/weights/best.pt --source 0 --show

# RealSense（另装相机依赖，WSL 需配置 USB 透传）
python -m pip install -r requirements-realsense.txt
bash run.sh detect --weights runs/segment/mango/weights/best.pt --source realsense --show
```

可加 `--device 0` 使用 GPU，`--max-frames 100` 限制处理帧数。
按 Q 或 Ctrl+C 结束实时运行。无桌面的环境不要加 `--show`。
当前 `--save-video` 用于视频和普通 USB 相机；RealSense 输出最新观测 JSON。

默认单图输出 `outputs/annotated.jpg`、`annotated.json`、`latest.json`。
视频/普通相机输出 `observations.jsonl` 保存逐帧结果；加 `--save-video` 输出 `annotated.mp4`。
流结束时 `latest.json` 会失效，视频最后的有效结果从 JSONL 读取。
每次流运行会覆盖该输出目录的历史文件，不同任务可指定不同 `--output`。

JSON schema_version=2，关键字段：

- `fruits`：每个果实的颜色、中心、`height_px`、置信度和可用深度。
- `decision.status`：confirmed 或 uncertain；confirmed 且目标为空表示已确认没有小果。
- `decision.small_id`：唯一小果 ID 或 null。
- `decision.small_ids`：长度为 0 或 1；不会包含两个小果。
- `decision.small_by_side`、`pairs`、`ratios`：侧别、各侧高度及比值。
- `decision.size_labels`：small、large 或 equal；不确定时为空。
- `decision.targets`：默认只含选中的小果。
- `temporal_stable`：连续帧一致；单图恒为 false。
- `timestamp_unix_s`：读取最新观测时检查是否过期。

ID 是当前帧内的检测编号，连续帧稳定判断使用左右上下槽位匹配。
RealSense 输出的 `surface_xyz_m` 是相机坐标系中的可见表面点，单位米。
它不是果实中心或机械臂抓取位姿；项目仍保留原有的观测输出范围，不包含机械臂驱动。

## 5. 无数据集验证

```bash
python3 demo.py
python3 -m unittest discover -s tests -v
# 安装依赖后也可使用 bash run.sh demo / bash run.sh test
```

demo 使用手工构造的高度/颜色，输出 `outputs/demo.json`，不证明真实识别准确率。
测试包含一个“较矮果面积反而更大”的场景，确保最终选果只依赖高度。
图像/视频集成测试需要 OpenCV 和 numpy，缺少时明确跳过。

接口参考：[Ultralytics 推理](https://docs.ultralytics.com/modes/predict/)、[验证](https://docs.ultralytics.com/modes/val/)。
也支持 Conda：先创建并激活 Python 3.10 环境，运行 `bash setup_conda.sh`。
完整步骤见 [TRAINING.md 的 Conda 说明](TRAINING.md#使用-conda可替代-setup_wslsh)。