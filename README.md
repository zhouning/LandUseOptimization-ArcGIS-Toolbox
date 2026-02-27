# LandUseOptimization-ArcGIS-Toolbox

基于深度强化学习（DRL）的耕地空间布局优化 ArcGIS 工具箱。

ArcGIS Toolbox for Farmland Layout Optimization using Deep Reinforcement Learning.

---

## 功能简介

通过训练好的评分网络，执行配对的耕地-林地用地互换，在**保持耕地总量不变**（FC=0）的前提下：

- **降低耕地平均坡度** — 将高坡度耕地置换为林地
- **提升空间连片性** — 使耕地在空间上更加集中连片

## 两个版本

本仓库提供两个版本的工具箱，适配不同的 ArcGIS 平台：

| 版本 | 目录 | 平台 | Python | 推理引擎 | 额外依赖 |
|------|------|------|--------|---------|---------|
| **ArcGIS Pro 版** | `arcgis_toolbox_pro/` | ArcGIS Pro 3.x+ | Python 3.9+ (64-bit) | PyTorch | 需安装 PyTorch CPU |
| **ArcMap 版** | `arcgis_toolbox_arcmap/` | ArcMap 10.2+ | Python 2.7 (32-bit) | 纯 NumPy | 无（开箱即用） |

> 两个版本的**优化结果完全一致**（NumPy 推理与 PyTorch 推理误差 < 1e-5）。

## 快速开始

### ArcGIS Pro 用户

1. 安装 PyTorch CPU 版（ArcGIS Pro Python Command Prompt）：
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
2. ArcGIS Pro → Catalog → Toolboxes → Add Toolbox → 选择 `arcgis_toolbox_pro/LandUseOptimization.pyt`
3. 运行 **Check Dependencies** 验证环境
4. 运行 **DRL Land Use Optimization** 进行优化

### ArcMap 用户

1. **无需安装任何依赖**
2. ArcMap → ArcToolbox → 右键 → Add Toolbox → 选择 `arcgis_toolbox_arcmap/LandUseOptimization.pyt`
3. 运行 **Check Dependencies** 验证环境
4. 运行 **DRL Land Use Optimization** 进行优化

## 输入数据要求

- **几何类型**：面（Polygon）
- **坐标系**：推荐投影坐标系（单位为米）
- **必需字段**：
  - 用地分类字段（Text）— 如 `DLMC`，值如"旱地""水田""果园""有林地"
  - 坡度字段（Double/Single）— 如 `Slope`

## 输出字段

| 字段 | 类型 | 说明 |
|------|------|------|
| OPT_DLMC | Text | 优化后用地分类名称 |
| OPT_TYPE | Short | 0=其他, 1=耕地, 2=林地 |
| CHG_FLAG | Short | 0=未变, 1=耕地→林地, 2=林地→耕地 |
| ORIG_DLMC | Text | 原始用地分类名称 |

## 目录结构

```
├── arcgis_toolbox_pro/                  ← ArcGIS Pro 版本
│   ├── LandUseOptimization.pyt          ← 主工具箱 (Python 3)
│   ├── *.pyt.xml                        ← 帮助元数据
│   ├── core/                            ← 核心模块 (PyTorch)
│   ├── models/
│   │   └── scorer_weights_v7.pt         ← 模型权重 (PyTorch)
│   └── 用户手册_耕地空间布局优化工具箱.md
│
├── arcgis_toolbox_arcmap/               ← ArcMap 版本
│   ├── LandUseOptimization.pyt          ← 主工具箱 (Python 2.7)
│   ├── *.pyt.xml                        ← 帮助元数据
│   ├── core/                            ← 核心模块 (纯 NumPy)
│   ├── models/
│   │   └── scorer_weights_v7.npz        ← 模型权重 (NumPy)
│   └── 部署与使用手册_ArcMap版.md
│
└── convert_weights_to_npz.py            ← 权重格式转换脚本 (.pt → .npz)
```

## 文档

- **ArcGIS Pro 版**：[用户手册](arcgis_toolbox_pro/用户手册_耕地空间布局优化工具箱.md)
- **ArcMap 版**：[部署与使用手册](arcgis_toolbox_arcmap/部署与使用手册_ArcMap版.md)

## 技术原理

- **算法**：基于深度强化学习训练的评分网络
- **推理策略**：配对推理（交替 耕地→林地 / 林地→耕地），保证耕地总量守恒（FC=0）
- **核心模块以预编译形式分发**，详细方法请参见相关学术论文（有兴趣可以联系：zhouning@stu.pku.edu.cn）

## License

MIT
