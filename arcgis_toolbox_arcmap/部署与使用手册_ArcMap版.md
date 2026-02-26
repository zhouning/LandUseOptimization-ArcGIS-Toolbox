# 耕地空间布局优化工具箱 (DRL) — ArcMap 版部署与使用手册

---

## 一、工具简介

本工具箱基于**深度强化学习（Maskable PPO）**技术，实现耕地空间布局优化。通过配对的耕地-林地用地互换，在**保持耕地总量不变**的前提下：

- 降低耕地平均坡度（将高坡度耕地置换为林地）
- 提升耕地空间连片性（使耕地分布更加集中）

本版本专为 **ArcMap 10.x** 设计，使用纯 NumPy 实现模型推理，**无需安装 PyTorch 等额外依赖**，开箱即用。

### 与 ArcGIS Pro 版本的差异

| 对比项 | ArcGIS Pro 版 | ArcMap 版（本手册） |
|--------|--------------|-------------------|
| Python 环境 | Python 3.9+（64 位） | Python 2.7（32 位） |
| 推理引擎 | PyTorch | 纯 NumPy |
| 模型权重格式 | `.pt` | `.npz` |
| 额外依赖 | 需安装 PyTorch | 无（NumPy 已内置） |
| 优化结果 | — | 与 Pro 版完全一致 |

---

## 二、系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 7/10/11（64 位） |
| GIS 软件 | ArcMap 10.2 及以上（已验证 10.7.1） |
| Python | ArcMap 自带 Python 2.7（无需额外安装） |
| 许可级别 | Standard 或 Advanced（推荐 Advanced） |
| 磁盘空间 | 工具箱约 200 KB |

> **关于许可级别**：
> - **Advanced（ArcInfo）**许可：使用 `PolygonNeighbors` 工具快速构建空间邻接图，速度快
> - **Standard** 许可：自动回退到几何相交方法，速度较慢但**结果完全一致**

---

## 三、部署步骤

### 3.1 获取工具箱文件

将 `arcgis_toolbox_arcmap` 文件夹整体复制到目标机器的任意目录，例如：

```
D:\GIS_Tools\arcgis_toolbox_arcmap\
```

文件夹内容如下，**请勿改变内部结构或文件相对位置**：

```
arcgis_toolbox_arcmap\
│
├── LandUseOptimization.pyt                              ← 主工具箱文件
├── LandUseOptimization.pyt.xml                          ← 工具箱帮助信息
├── LandUseOptimization.CheckDependenciesTool.pyt.xml    ← 依赖检查工具帮助
├── LandUseOptimization.OptimizeLandUseTool.pyt.xml      ← 优化工具帮助
│
├── core\                                                ← 核心计算模块
│   ├── __init__.py
│   ├── scorer_standalone.py    ← DRL 评分网络（纯 NumPy）
│   ├── data_io.py              ← 数据读写
│   ├── adjacency.py            ← 空间邻接图构建
│   └── paired_inference.py     ← 配对推理引擎
│
└── models\
    └── scorer_weights_v7.npz   ← 训练好的模型权重（NumPy 格式）
```

### 3.2 在 ArcMap 中加载工具箱

1. 打开 ArcMap，加载或新建一个地图文档（.mxd）

2. 打开 **ArcToolbox** 窗口（菜单栏 → Geoprocessing → ArcToolbox，或点击工具栏上的红色工具箱图标）

3. 在 ArcToolbox 面板中**右键单击空白处**，选择 **Add Toolbox...**

4. 浏览到 `arcgis_toolbox_arcmap` 文件夹，选中 `LandUseOptimization.pyt`，点击 **Open**

5. 工具箱出现在列表中，展开可看到两个工具：
   - **Check Dependencies**（依赖检查）
   - **DRL Land Use Optimization**（DRL 用地优化）

> **提示**：如希望工具箱在每次启动 ArcMap 时自动可用，可右键 ArcToolbox 面板根节点 → **Save Settings** → **To Default**

### 3.3 验证环境（首次使用必做）

1. 展开工具箱，双击 **Check Dependencies**
2. 直接点击 **OK** 运行（无需设置参数）
3. 查看消息窗口输出，确认所有检查项显示 OK：

```
=== Dependency Check (ArcMap) ===
Python: 2.7.16 (...)
NumPy: 1.9.3 (OK)
arcpy: 10.7.1 (OK)
License: ArcInfo
PolygonNeighbors: Available (Advanced license)
Model weights: ...\models\scorer_weights_v7.npz (42 KB, OK)

Note: This ArcMap version uses pure NumPy for model inference.
PyTorch is NOT required.

All dependencies are satisfied. Ready to use!
```

如有红色错误提示，请根据消息内容排查。

---

## 四、使用方法

### 4.1 准备输入数据

**数据要求：**

| 项目 | 要求 |
|------|------|
| 几何类型 | 面（Polygon） |
| 文件格式 | Shapefile (.shp)、文件地理数据库 (.gdb)、个人地理数据库 (.mdb) |
| 坐标系 | 推荐投影坐标系（单位为米），如 CGCS2000 高斯-克吕格投影 |

**必需字段：**

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| 用地分类字段 | Text | 地块的用地分类名称 | DLMC（值如"旱地""水田""果园"等） |
| 坡度字段 | Double/Single/Long/Short | 地块的坡度值（度） | Slope |

**重要提示：**
- 用地分类字段中的文本值须与工具参数中设置的耕地/林地类型名称**完全一致**（包括全角/半角、空格等）
- 建议提前查看属性表，确认分类名称的准确写法

### 4.2 运行优化

1. 展开工具箱，双击 **DRL Land Use Optimization**

2. 在弹出的工具对话框中设置参数：

   **必需参数：**

   | 参数 | 操作 |
   |------|------|
   | **Input Feature Class** | 点击文件夹图标浏览选择输入面数据 |
   | **Land Use Classification Field (DLMC)** | 从下拉列表中选择用地分类字段 |
   | **Slope Field** | 从下拉列表中选择坡度字段 |
   | **Output Feature Class** | 指定输出结果保存路径和文件名 |

   **可选参数（通常保持默认即可）：**

   | 参数 | 默认值 | 说明 |
   |------|--------|------|
   | **Model Weights File** | models/scorer_weights_v7.npz | 模型权重，一般无需修改 |
   | **Number of Conversion Pairs** | 100 | 互换对数（1-500），越大优化越充分 |
   | **Farmland Type Names** | 旱地; 水田 | 耕地分类名称，按数据实际值修改 |
   | **Forest Type Names** | 果园; 有林地 | 林地分类名称，按数据实际值修改 |

3. 点击 **OK** 开始运行

4. 运行过程中，进度条和消息窗口会实时显示处理进度：
   - 阶段 1：加载模型权重
   - 阶段 2：读取数据、分类地块
   - 阶段 3：构建空间邻接图
   - 阶段 4：执行配对推理（显示实时坡度和连片性指标）
   - 阶段 5：写入输出要素类

### 4.3 查看运行结果

运行完成后，消息窗口显示优化摘要，例如：

```
=== Optimization Results ===
  Completed pairs: 100
  Slope: 12.3456 -> 11.9234 (change: -0.4222, -3.42%)
  Contiguity: 2.1500 -> 2.2037 (change: +0.0537)
  Farmland count change: 0

Output: D:\results\optimized.shp
```

**结果解读：**

| 指标 | 含义 | 期望方向 |
|------|------|----------|
| Slope change | 耕地平均坡度变化量 | 负值（坡度降低） |
| Slope change % | 坡度变化百分比 | 负值 |
| Contiguity change | 空间连片性变化量 | 正值（连片性提升） |
| Farmland count change | 耕地数量变化 | 0（耕地总量守恒） |

---

## 五、输出结果说明

输出要素类保留了输入数据的所有原始字段和几何，并新增以下 4 个字段：

| 字段名 | 类型 | 含义 | 值说明 |
|--------|------|------|--------|
| **OPT_DLMC** | Text(30) | 优化后的用地分类名称 | 变化地块显示新分类名，未变化地块保持原值 |
| **OPT_TYPE** | Short | 优化后的类型编码 | 0 = 其他，1 = 耕地，2 = 林地 |
| **CHG_FLAG** | Short | 变化标记 | 0 = 未变化，1 = 耕地→林地，2 = 林地→耕地 |
| **ORIG_DLMC** | Text(30) | 原始用地分类名称 | 保留原始值，便于与优化结果对比 |

### 结果可视化建议

按 `CHG_FLAG` 字段进行**分类符号化**渲染：

| CHG_FLAG 值 | 含义 | 建议颜色 |
|-------------|------|----------|
| 0 | 未变化 | 灰色或透明 |
| 1 | 耕地 → 林地 | 绿色 |
| 2 | 林地 → 耕地 | 黄色/橙色 |

**操作步骤：**
1. 在 ArcMap 内容列表中右键输出图层 → **Properties** → **Symbology** 选项卡
2. 左侧选择 **Categories** → **Unique values**
3. Value Field 选择 `CHG_FLAG`
4. 点击 **Add All Values**
5. 分别双击各符号修改颜色 → **OK**

---

## 六、参数详细说明

### 6.1 输入要素类（Input Feature Class）

- 类型：面（Polygon）要素类
- 格式：Shapefile、文件地理数据库、个人地理数据库
- 要求：必须包含用地分类文本字段和坡度数值字段
- 坐标系：推荐使用投影坐标系（单位为米）

### 6.2 用地分类字段（Land Use Classification Field）

- 类型：文本字段
- 用途：工具通过该字段的文本值区分耕地、林地和其他地块
- **关键**：字段中的值必须与"耕地类型名称"和"林地类型名称"参数中的值**完全匹配**
- 不属于耕地或林地的地块自动归类为"其他"，不参与优化

### 6.3 坡度字段（Slope Field）

- 类型：数值字段（Double、Single、Long 或 Short）
- 单位：度
- 用途：优化的核心目标 — 降低耕地的平均坡度
- 来源：通常由 DEM 通过区域统计（Zonal Statistics）计算得到

### 6.4 输出要素类（Output Feature Class）

- 输出路径支持 Shapefile 或地理数据库格式
- 输出包含原始所有字段 + 4 个新增字段

### 6.5 模型权重文件（Model Weights File）

- 格式：`.npz`（NumPy 压缩数组格式）
- 默认值：工具箱自带的 `models/scorer_weights_v7.npz`
- 说明：模型是维度无关的，同一权重文件可用于不同规模的数据，无需重新训练

### 6.6 转换对数（Number of Conversion Pairs）

- 范围：1 - 500
- 默认：100
- 每一对包含一次耕地→林地转换和一次林地→耕地转换
- 耕地总量始终守恒（FC=0）
- 建议值：
  - **30 - 50**：快速测试，优化幅度较小
  - **100**（默认）：平衡的优化效果
  - **200 - 500**：更充分的优化，耗时更长
- 实际执行对数不会超过耕地和林地数量中的较小值

### 6.7 耕地类型名称 / 林地类型名称

- 类型：文本列表（多值）
- 默认值：
  - 耕地：`旱地`、`水田`
  - 林地：`果园`、`有林地`
- **必须与数据中的实际值完全一致**
- 可根据数据实际情况自行添加或修改

---

## 七、操作示例

以"斑竹村"数据集为例，演示完整操作流程。

### 7.1 步骤一：加载工具箱

1. 打开 ArcMap
2. ArcToolbox → 右键 → Add Toolbox → 选择 `LandUseOptimization.pyt`

### 7.2 步骤二：运行依赖检查

1. 双击 **Check Dependencies** → OK
2. 确认所有项显示 OK

### 7.3 步骤三：添加数据到地图

1. 将输入数据 `斑竹村10000.shp` 添加到地图中
2. 打开属性表确认字段：
   - `DLMC` 字段包含值：旱地、水田、果园、有林地 等
   - `Slope` 字段包含坡度数值

### 7.4 步骤四：运行优化

1. 双击 **DRL Land Use Optimization**
2. 参数设置：
   - Input Feature Class → `斑竹村10000.shp`
   - Land Use Classification Field → `DLMC`
   - Slope Field → `Slope`
   - Output Feature Class → `D:\results\banzhucun_optimized.shp`
   - 其他参数保持默认
3. 点击 OK，等待运行完成

### 7.5 步骤五：查看结果

1. 输出图层自动加载到地图
2. 右键图层 → Properties → Symbology
3. 选择 Categories → Unique values → CHG_FLAG
4. 设置颜色方案查看变化地块

---

## 八、常见问题

### Q1: 工具箱加载后看不到工具？

**A**：确认 `.pyt` 文件所在文件夹中 `core/` 子文件夹和 `models/` 子文件夹完整存在。如果仅复制了 `.pyt` 文件而遗漏了子文件夹，工具箱可能无法正确加载。

### Q2: 运行时提示 "No farmland parcels found"？

**A**：用地分类字段中没有找到与"耕地类型名称"匹配的值。请检查：
- 用地分类字段是否选择正确
- 打开属性表查看字段中的实际文本值
- 确保耕地类型名称参数与数据中的值**完全一致**（注意空格、全角半角）

### Q3: 运行时提示 "No forest parcels found"？

**A**：与 Q2 同理，检查林地类型名称参数是否与数据中的值匹配。

### Q4: 邻接图构建阶段非常慢？

**A**：可能原因：
- 许可级别为 Standard，使用了几何相交回退方法。升级到 Advanced 许可可大幅提速
- 数据量过大。32 位环境处理上万条记录时会较慢

### Q5: 运行中途出现内存错误？

**A**：ArcMap 使用 32 位 Python，内存上限约 2 GB。对于超过 50,000 条记录的大数据集：
- 可考虑裁剪研究区域，分区域处理
- 推荐使用 ArcGIS Pro 版本（64 位，无内存限制）

### Q6: 可以用于其他地区的数据吗？

**A**：可以。模型是维度无关的，同一权重文件适用于任意数量地块的数据集。但请确保：
- 数据包含用地分类（文本）和坡度（数值）字段
- 根据数据实际值调整耕地/林地类型名称参数

### Q7: 如何修改耕地/林地类型名称？

**A**：在工具对话框中：
1. 找到 **Farmland Type Names** 或 **Forest Type Names** 参数
2. 删除不需要的默认值，输入新的分类名称
3. 多个名称之间用分号分隔，或逐个添加

### Q8: 模型权重文件从哪里来？

**A**：工具箱已自带 `scorer_weights_v7.npz` 权重文件。如需更新权重：
1. 在 Python 3 + PyTorch 环境下运行 `convert_weights_to_npz.py`
2. 将生成的 `.npz` 文件复制到 `models/` 文件夹
3. 运行优化时选择新的权重文件

### Q9: 输出的 Farmland count change 不为 0？

**A**：在配对推理模式下此值应始终为 0。如果不为 0，请检查：
- 耕地类型名称和林地类型名称是否有交叉（同一分类名同时出现在两个列表中）
- 如确认参数无误，请联系开发者排查

### Q10: 如何使用 Python 脚本调用工具？

**A**：在 ArcMap 的 Python 窗口中：

```python
import arcpy

# 加载工具箱
arcpy.ImportToolbox(r"D:\GIS_Tools\arcgis_toolbox_arcmap\LandUseOptimization.pyt")

# 运行优化
arcpy.LandUseOpt.OptimizeLandUseTool(
    input_fc=r"D:\data\study_area.shp",
    dlmc_field="DLMC",
    slope_field="Slope",
    output_fc=r"D:\results\optimized.shp",
    model_weights=r"D:\GIS_Tools\arcgis_toolbox_arcmap\models\scorer_weights_v7.npz",
    n_pairs=100,
    farmland_types=["旱地", "水田"],
    forest_types=["果园", "有林地"],
)
```

---

## 九、技术原理简介

### 算法框架

本工具采用 **Maskable PPO**（带动作掩码的近端策略优化）算法，训练了一个评分网络。该网络对每个地块进行评分，通过掩码机制选择最优的互换对象。

### 配对推理策略

每一轮操作包含两步：
1. **Phase 0**：从所有耕地中选出最优地块，将其转为林地
2. **Phase 1**：从所有林地中选出最优地块，将其转为耕地

每完成一对操作，耕地总量保持不变，实现 **FC=0**（耕地数量守恒）。

### 纯 NumPy 推理

ArcMap 版本将神经网络的前向传播拆解为纯 NumPy 矩阵运算，无需 PyTorch 依赖。经严格验证，与 PyTorch 版本的输出误差小于 0.000004，优化结果完全一致。

详细的算法设计与特征工程请参见相关学术论文。
