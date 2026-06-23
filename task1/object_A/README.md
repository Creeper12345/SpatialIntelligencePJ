# object_A: 多视角真实物体重建

物体 A 是真实拍摄的绒毛鸟玩具。流程为视频抽帧、前景分割、COLMAP 位姿估计，再使用 3D Gaussian Splatting 训练物体点云。

## 内容

- `colmap/`: COLMAP dense reconstruction 脚本。
- `data/images/`: 从视频抽取的多视角输入帧。
- `data/images_masked/`: 前景分割后的 masked 输入帧。
- `data/mask_preview.png`: 分割效果预览。
- `results/gs_output_masked/`: masked 数据训练得到的 3DGS 小型输出、metadata、orbit frames/video 和日志。
- `results_preview/`: 报告中使用的物体 A 预览图。

## 未包含

COLMAP `distorted/database.db` 和更大的中间缓存未放入提交目录。
