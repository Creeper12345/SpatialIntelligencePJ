# object_C: 单图到 3D 资产生成

物体 C 是使用单张咖啡罐 RGBA 图片，通过 threestudio + Zero123 生成的 3D 资产。

## 内容

- `configs/`: Zero123 训练命令和配置。
- `results/stable_lowmem_save/`: lowmem 版本最终视频、训练图和 `it600-export/` mesh；融合脚本默认使用该版本的 mesh。
- `results/stable_final_save/`: final 版本最终视频、训练图和 `it3701-export/` mesh。
- `results_preview/`: 输入图和多视角重建预览图。
- `external_repos/threestudio/`: 第三方 threestudio 仓库占位目录，源码未复制。

