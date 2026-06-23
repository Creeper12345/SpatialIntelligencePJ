# background: 背景场景重建

背景场景使用 Mip-NeRF 360 数据集中的 `garden` 场景，并以 3D Gaussian Splatting 表示作为融合背景。

## 内容

- `results/cfg_args`: 背景 3DGS 配置参数。
- `results/cameras.json`: 背景相机 metadata。
- `results/exposure.json`: 曝光 metadata。
- `results/input.ply`: 背景输入稀疏点云。
- `results/comparison.png`: 背景重建比较图。
- `results/*.log`: 背景训练/渲染日志。
- `external_repos/gaussian-splatting/`: 第三方 3DGS 仓库占位目录，源码未复制。

## 未包含

Mip-NeRF 360 原始 garden 数据、背景完整 3DGS 点云和 checkpoint 未放入提交目录；完整点云单文件接近 900MB。
