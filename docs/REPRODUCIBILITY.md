# Reproducibility Guide: LLM新闻分析实验

## 摘要

本文档详细描述了如何一键复现LLM新闻分析实验的所有结果。

---

## 快速开始（5分钟）

### 方法1：使用Python脚本（推荐，跨平台）

```bash
# 一键复现所有实验
python scripts/reproduce_all.py
```

### 方法2：Windows批处理文件

```bash
# Windows用户
scripts\reproduce_all.bat
```

---

## 详细复现步骤

### 1. 环境设置

#### 1.1 软件要求

- Python: 3.8 或更高版本
- Git: 可选（用于版本控制）
- 磁盘空间: 至少 2GB

#### 1.2 安装依赖

```bash
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 1.3 配置环境变量（可选）

```bash
# 复制.env.example为.env
cp .env.example .env

# 编辑.env（如果需要使用真实LLM API）
# 填入你的API key
```

### 2. 数据准备

#### 2.1 检查数据

```bash
# 检查From_News_to_Forecast项目是否存在
ls -la From_News_to_Forecast/
```

#### 2.2 数据来源

所有数据已包含在项目中：

- **新闻数据**：From_News_to_Forecast/data/raw_news_data/bitcoin_news.json
- **价格数据**：通过yfinance自动下载
- **来源**：From_News_to_Forecast（NeurIPS 2024）

### 3. 一键复现

#### 3.1 运行复现脚本

```bash
# 一键复现所有实验
python scripts/reproduce_all.py
```

#### 3.2 预期输出

```
================================================================================
LLM新闻分析实验 - 一键复现脚本
================================================================================

随机种子: 42
开始时间: 2026-04-06 07:15:00

[2026-04-06 07:15:00] 开始复现所有实验...

[2026-04-06 07:15:00] 开始实验: Sanity Check
[2026-04-06 07:15:01] ================================================================
[2026-04-06 07:15:01] Experiment 1: Sanity Check
[2026-04-06 07:15:01] ================================================================
...
[2026-04-06 07:15:30] ✓ 实验完成: Sanity Check

[2026-04-06 07:15:30] 开始实验: Final Experiment (AI vs Rule)
...
[2026-04-06 07:16:00] ✓ 实验完成: Final Experiment (AI vs Rule)

================================================================================
复现总结
================================================================================
成功实验: 2/2
结束时间: 2026-04-06 07:16:00
日志文件: results/logs/reproduce_log_20260406_071500.txt

✓ 所有实验复现成功！

================================================================================
Reproduction completed successfully
================================================================================
```

### 4. 查看结果

#### 4.1 结果位置

所有结果保存在 `results/` 目录：

```
results/
├── tables/                    # 结果表格
│   ├── 01_sanity_check_results.csv
│   ├── final_experiment_real_data_results.csv
│   └── ...
├── figures/                   # 可视化图表
└── logs/                     # 实验日志
    └── reproduce_log_YYYYMMDD_HHMMSS.txt
```

#### 4.2 主要结果

**Sanity Check结果：**
- Perfect策略：显著盈利（p-value < 0.05）
- Random策略：接近0收益
- ✅ 验证通过

**最终实验结果：**
- LLM策略 vs Rule策略：p-value > 0.05
- ✅ 无显著差异

### 5. 故障排除

#### 5.1 常见问题

**问题1：找不到From_News_to_Forecast项目不存在**

```
错误: FileNotFoundError: From_News_to_Forecast/...

解决方法:
1. 确认From_News_to_Forecast目录在项目根目录
2. 如果没有，请从GitHub克隆:
   git clone https://github.com/ameliawong1996/From_News_to_Forecast.git
```

**问题2：缺少Python依赖**

```
错误: ModuleNotFoundError: No module named 'xxx'

解决方法:
1. 确保已安装所有依赖:
   pip install -r requirements.txt
```

**问题3：yfinance下载失败**

```
错误: Failed to download price data

解决方法:
1. 检查网络连接
2. 确保可以访问Yahoo Finance
3. 或者使用离线价格数据（如果有）
```

#### 5.2 可复现性检查清单

- [ ] 所有随机种子已固定（seed=42）
- [ ] 无数据泄露（时间顺序严格验证）
- [ ] 所有实验都有统计检验
- [ ] 所有结果可复现
- [ ] 所有变量命名清晰
- [ ] 代码可直接运行

### 6. 手动复现（可选）

如果您希望手动运行每个实验：

#### 6.1 运行Sanity Check

```bash
python experiments/01_sanity_check.py
```

#### 6.2 运行最终实验

```bash
python experiments/final_experiment.py
```

### 7. 可复现性保证

#### 7.1 随机种子

- 主随机种子：42
- 所有随机操作使用固定种子
- PYTHONHASHSEED=42

#### 7.2 数据固定

- 使用From_News_to_Forecast的真实数据
- 价格数据通过yfinance下载（可复现）
- 采样方法固定（均匀采样）

#### 7.3 代码固定

- 所有代码版本控制
- 模块化设计，清晰的模块划分
- 完整的注释

### 8. 引用

如果您复现了本实验，请引用：

```bibtex
@inproceedings{wang2024news,
  title={From News to Forecast: Integrating Event Analysis in LLM-based Time Series Forecasting with Reflection},
  author={Wang, Xinlei and Feng, Maike and Qiu, Jing and Gu, Jinjin and Zhao, Junhua},
  booktitle={NeurIPS},
  year={2024}
}
```

---

**文档版本：** 1.0  
**最后更新：** 2026-04-06  
**维护者：** LLM新闻分析实验团队
