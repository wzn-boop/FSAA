markdown
# FSAA Causal Discovery System

基于特征增强和非对称性的因果发现系统，集成FSAA特征选择、格兰杰因果分析和结构方程建模，用于异常检测和根因分析。

## 核心特性

- **FSAA特征选择**: 基于信息熵和非对称性准则，筛选保留因果关系的关键变量
- **特征增强**: 自动创建滞后特征和非线性变换特征
- **动态因果发现**: 结合格兰杰因果检验和结构方程建模
- **异常检测**: 通过对比正常/异常时段的因果图结构变化
- **根因定位**: 识别引发系统异常的关键维度

## 系统架构
FSAA Causal Discovery System
├── Feature Augmentation Layer # 特征增强层
├── FSAA Selection Layer # 特征选择层
├── Causal Discovery Layer # 因果发现层
└── Anomaly Detection Layer # 异常检测层

text

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```
运行示例
bash
python main.py \
  --normal_data data/normal_period.csv \
  --anomaly_data data/anomaly_period.csv \
  --target target_variable \
  --output results \
  --threshold 0.1 \
  --max_lag 3
命令行参数
参数	说明	默认值
--normal_data	正常时段数据文件路径	必需
--anomaly_data	异常时段数据文件路径	必需
--target	目标变量列名（可选）	None
--output	输出目录路径	results
--threshold	FSAA特征选择阈值	0.1
--max_lag	最大滞后阶数	3
🔧 核心算法
FSAA特征选择
FSAA（Feature Selection based on Augmentation and Asymmetry）方法：

特征增强: 生成滞后、差分、非线性变换特征

信息熵计算: 评估特征的信息含量

因果非对称性: 计算特征间的因果方向强度

特征筛选: 基于综合评分选择关键变量

因果发现流程
正常模式学习:

FSAA特征选择 → 格兰杰因果检验 → SEM优化 → 构建因果图

异常检测:

构建异常因果图 → 对比结构变化 → 定位异常源

输出结果
系统生成以下输出文件：

results/fsaa_results.png: 综合可视化结果

results/feature_importance.csv: 特征重要性排序

results/anomaly_sources.csv: 异常源分析结果

results/analysis_report.txt: 详细分析报告