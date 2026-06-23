# fusion: 物体与背景融合、渲染

本模块包含自写脚本，用于将物体 A 的 3DGS 点云和物体 B/C 的 mesh 统一转换到高斯球表示，拼接进 garden 背景并渲染漫游视频。

## 内容

- `code/`: 融合、mesh-to-Gaussian 转换和 3DGS 渲染脚本。
- `results/scene_fusion.mp4`: 最终融合场景视频。
- `results/orbit_frames/`: 最终融合视频的帧序列。
- `results/cameras.json`, `results/cfg_args`: 融合场景 metadata。
- `results_preview/`: 报告中使用的融合场景预览图。
- `external_repos/gaussian-splatting/`: 第三方 3DGS 仓库占位目录，源码未复制。

## 未包含

融合后的完整 PLY 点云未放入提交目录；`fused_scene/point_cloud/iteration_99999/point_cloud.ply` 约 975MB，不能直接上传 GitHub。
