# Task 1 提交目录

本目录按 Task 1 的资产生成与融合流程整理为五个模块，并补充了可直接放入 GitHub 的数据和结果文件。

```text
task1/
├── object_A/              # 多视角真实物体重建
├── object_B/              # 文本到 3D 资产生成
├── object_C/              # 单图到 3D 资产生成
├── background/            # 背景场景重建
└── fusion/                # 物体与背景融合、渲染
```

当前目录约 526MB，未包含超过 50MB 的单文件，也没有超过 GitHub 100MB 硬限制的文件。

## 模块说明

- `object_A/`: 真实多视角绒毛鸟玩具重建。包含 COLMAP 脚本、输入帧、masked 帧、最终 3DGS 小型输出和预览图。
- `object_B/`: 文本到 3D 篮球资产生成。包含 threestudio/DreamFusion 配置、训练结果 `save/`、导出 mesh、测试视频/帧和预览图。
- `object_C/`: 单图到 3D 咖啡罐资产生成。包含 threestudio/Zero123 配置、final/lowmem 导出 mesh、最终视频和预览图。
- `background/`: Mip-NeRF 360 garden 背景场景。包含小型 metadata、日志、输入 PLY 和比较图；不含超大背景点云。
- `fusion/`: 包含融合/渲染脚本、最终融合视频和 orbit frames；不含超大融合点云。

## 外部依赖

- `gaussian-splatting`: 用于物体 A、背景 3DGS 训练/加载，以及融合场景渲染。
- `threestudio`: 用于物体 B DreamFusion 和物体 C Zero123 训练、导出 mesh。
- `COLMAP`: 用于物体 A 多视角相机位姿估计和稠密重建。

第三方仓库源码仍只保留占位目录，不复制完整仓库。

## 仍未包含的大文件

- `*.ckpt` checkpoint，单文件约 151MB，超过 GitHub 普通文件硬限制。
- COLMAP `database.db`，约 52MB，虽然可上传但不适合 Git 仓库。
- 背景 3DGS 点云，例如 `output/garden/point_cloud/.../point_cloud.ply`，约 789-899MB。
- 融合后的完整 3DGS 点云 `fused_scene/point_cloud/iteration_99999/point_cloud.ply`，约 975MB。
- Mip-NeRF 360 `garden` 原始数据集，约 5.7GB。
- threestudio 自动复制的完整第三方 `code/` 快照。
