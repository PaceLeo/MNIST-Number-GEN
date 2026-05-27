# 256x256 手写数字条件 GAN

本项目用于生成 `256x256` 手写数字。当前版本只保留三个控制条件：

- 数字类别：`0` 到 `9`
- 笔画粗细：`thickness`，范围 `[0, 1]`
- 倾斜度：`slant`，范围 `[0, 1]`

writer/style 控制已经删除。旧的 128x128 兼容逻辑也已经删除，当前代码只面向重新训练 256x256 模型。

## 项目结构

```text
.
|-- data/
|   |-- Images.npy                 # 原始 500x500 图像
|   |-- WriterInfo.npy             # 原始标签
|   |-- Images_256.npy             # 预处理后的 256x256 图像
|   |-- WriterInfo_256.npy         # 与 Images_256.npy 对齐的标签
|   `-- PreprocessInfo_256.npz     # 粗细、倾斜度等元数据
|-- cgan-images-256-clean/         # 新模型生成图输出目录
|-- cgan_model.py                  # 256-only 生成器和判别器
|-- preprocess_digits.py           # 数据预处理脚本
|-- train.py                       # 训练脚本
|-- gen.py                         # 生成脚本
|-- control_sweep.py               # 粗细/倾斜度扫描脚本
|-- clean_generated.py             # 展示用后处理脚本
`-- requirements.txt
```

## 环境

当前使用 conda 环境 `gan`。

检查 PyTorch 和 CUDA：

```powershell
conda run -n gan python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 数据预处理

当前已经存在 256 数据：

- `data/Images_256.npy`
- `data/WriterInfo_256.npy`
- `data/PreprocessInfo_256.npz`

如需重新生成：

```powershell
conda run -n gan python preprocess_digits.py --out-images data/Images_256.npy --out-writer-info data/WriterInfo_256.npy --out-meta data/PreprocessInfo_256.npz --output-size 256 --content-size 224
```

## 训练

推荐从头训练：

```powershell
conda run -n gan python train.py --epochs 80 --batch-size 16 --base-channels 48 --sample-every 500 --save-every 5 --out-dir checkpoints_256_clean --sample-dir cgan-images-256-clean --device cuda
```

显存不足时：

```powershell
conda run -n gan python train.py --epochs 80 --batch-size 8 --base-channels 32 --sample-every 500 --save-every 5 --out-dir checkpoints_256_clean --sample-dir cgan-images-256-clean --device cuda
```

当前训练配置为了让输出更干净，默认做了这些改动：

- 真实训练图像默认二值化后再送入判别器。
- 真实训练图像默认做随机粗细增强，并把增强参数作为 `thickness` 条件。
- 真实训练图像默认做随机倾斜增强，并把增强参数作为 `slant` 条件。
- 生成器使用 `resize + conv`，避免转置卷积的棋盘纹理。
- 删除 writer/style 条件，减少模型需要学习的无效控制。
- 加入 `binary_loss`，压低灰边和中间灰度。
- 加入 `background_loss`，减少背景脏纹理。
- 日志会输出 `acc` 和 `ink`，比单看 GAN loss 更有用。

如果生成图过于硬、边缘太像二值图，可以降低：

```text
--binary-loss-weight 0.4
```

如果粗细控制不明显，可以提高粗细增强和粗细回归权重：

```text
--thickness-steps 5 --thickness-loss-weight 6
```

如果笔画过粗、细节被糊掉，则降低：

```text
--thickness-steps 3
```

如果背景仍然脏，可以提高：

```text
--background-loss-weight 2.0
```

如果倾斜度控制仍然不明显，可以提高倾斜增强和倾斜回归权重：

```text
--slant-strength 0.6 --slant-loss-weight 6
```

如果数字被拉得太变形，则降低：

```text
--slant-strength 0.35
```

## 生成

训练完成后生成数字：

```powershell
conda run -n gan python gen.py --checkpoint checkpoints_256_clean/latest.pt --digit 3 --num 16 --nrow 8 --thickness 0.6 --slant 0.5 --device cuda
```

生成并保存清理后的黑白展示图：

```powershell
conda run -n gan python gen.py --checkpoint checkpoints_256_clean/latest.pt --digit 3 --num 16 --nrow 8 --thickness 0.6 --slant 0.5 --clean --device cuda
```

默认文件名固定为 `digit{数字}.png`。例如 `--digit 3` 会保存为 `cgan-images-256-clean/digit3.png`；加 `--clean` 会保存为 `cgan-images-256-clean/digit3_clean.png`。如需自定义路径，再传 `--out`。

多组粗细/倾斜度示例：

```powershell
conda run -n gan python gen.py --checkpoint checkpoints_256_clean/latest.pt --digit 8 --num 8 --nrow 4 --thickness 0.2,0.4,0.6,0.8,0.2,0.4,0.6,0.8 --slant 0.2,0.2,0.2,0.2,0.8,0.8,0.8,0.8 --out cgan-images-256-clean/digit8_control.png --device cuda
```

## 控制变量检查

训练后检查粗细和倾斜度是否仍然有效：

```powershell
conda run -n gan python control_sweep.py --checkpoint checkpoints_256_clean/latest.pt --digit 3 --out cgan-images-256-clean/control_sweep_digit3.png --device cuda
```

## 备注

GAN 的 `d_loss` 和 `g_loss` 不需要接近 0。更应该看：

- 样例图是否越来越清楚
- `acc` 是否升高
- `ink` 是否接近真实图像的黑色像素比例
- 背景是否干净
- 粗细/倾斜度扫描是否单调
