# LLM News Analysis Experiment System
# LLM新闻分析实验系统

---

## English Version

### Project Overview

This repository contains the code for evaluating LLM-based financial news analysis strategies. Our research addresses the fundamental question: **Can large language models (LLMs) extract meaningful predictive signals from financial news for trading purposes?**

#### Research Questions
- **RQ1**: Do LLM strategies really outperform traditional strategies?
- **RQ2**: Which strategies perform best under what market conditions?
- **RQ3**: What are the advantages and disadvantages of different strategies?

### Core Contributions

- **Rigorous Statistical Evaluation**: T-tests, Wilcoxon tests, and bootstrap confidence intervals with Bonferroni correction
- **T+1 Trading Protocol**: Strict temporal alignment to avoid look-ahead bias
- **6 Strategies Compared**: LLM, Keyword, Hash, Random, Momentum, MeanReversion
- **Multi-Asset Validation**: Evaluation across stocks (AAPL, MSFT, GOOGL) and cryptocurrency (BTC-USD)
- **Comprehensive Analysis**: Error analysis, market condition analysis, sensitivity analysis
- **Publication-ready Visualizations**: Strategy comparison charts, cumulative returns, heatmaps
- **Complete Reproducibility**: One-click reproduction with fixed random seeds and detailed logging

### Project Structure

```
LLM-news/
├── src/                              # Core modules
│   ├── dataset_adapter.py             # Data financialization (From_News_to_Forecast)
│   ├── strategies.py                  # Unified strategy implementations
│   ├── strategy_momentum.py           # Momentum strategy
│   ├── strategy_meanreversion.py      # Mean Reversion strategy
│   ├── llm_model.py                  # LLM integration with caching
│   ├── backtest.py                   # Backtesting engine
│   ├── enhanced_stats.py              # Enhanced statistical tests
│   ├── error_analysis.py             # Error analysis module
│   ├── market_condition_analysis.py  # Market condition analysis
│   ├── sensitivity_analysis.py        # Sensitivity analysis
│   ├── visualization.py               # Visualization manager
│   └── experiment_logger.py           # Experiment logging
├── experiments/                       # Experiment scripts
│   ├── final_experiment.py           # Final experiment with all analyses
│   └── sanity_check.py               # Sanity check
├── scripts/                           # Utility scripts
│   ├── reproduce_all.py              # One-click reproduction
│   ├── generate_single_asset_results.py  # Single asset results
│   └── check_price_data.py          # Data quality check
├── docs/                              # Documentation
│   ├── REPRODUCIBILITY.md           # Reproducibility guide
│   ├── DATASET_CARD.md              # Dataset card
│   └── EXPERIMENTAL_PROTOCOL.md     # Experimental protocol
├── data/                              # Data directory
│   ├── cache/                        # Cached price data
│   └── From_News_to_Forecast/       # NeurIPS 2024 data
├── results/                           # Results directory
│   ├── tables/                       # Result tables
│   └── figures/                      # Visualization figures
├── config/                            # Configuration files
│   └── prompts.yaml                  # Prompt templates
├── requirements.txt                   # Python dependencies
├── .env.example                      # Environment variables template
└── README.md                         # This file
```

### Quick Start (3 Steps)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Configure environment (optional)
cp .env.example .env
# Edit .env if you want to use real LLM API

# Step 3: Run all experiments
python scripts/reproduce_all.py
```

### One-Click Reproduction

```bash
# Reproduce all experiments with one command
python scripts/reproduce_all.py

# Windows users
scripts\reproduce_all.bat
```

**Output**: All results saved to `results/` directory, including:
- Strategy comparison results
- Statistical tests and confidence intervals
- Error analysis, market condition analysis, sensitivity analysis
- Publication-ready visualizations
- Experiment metadata for reproducibility

### Data Overview

| Aspect | Description |
|--------|-------------|
| **News Data** | Real financial news from Reuters, Bloomberg, CNBC, Yahoo Finance |
| **Source** | From_News_to_Forecast (NeurIPS 2024) |
| **Price Data** | Yahoo Finance (via yfinance) |
| **Assets** | AAPL, MSFT, GOOGL, BTC-USD |
| **Time Period** | 2018-01-01 to 2021-05-11 |
| **Sample Size** | 150 news per asset |

### Experimental Results

#### Overall Performance (BTC-USD)

| Strategy | Return | Sharpe | Win Rate | Max Drawdown | Calmar | Sortino |
|----------|--------|--------|----------|--------------|--------|---------|
| **Keyword** | 0.0038 | 0.193 | 52.67% | 321.99% | 0.0012 | 0.368 |
| **Hash** | -0.0005 | -0.026 | 46.67% | 47950.22% | -0.0000 | -0.036 |
| **Random** | -0.0017 | -0.085 | 48.00% | -0.00% | 0.0000 | -0.096 |
| **Momentum** | -0.0019 | -0.093 | 46.00% | 646.15% | -0.0003 | -0.107 |
| **MeanReversion** | -0.0012 | -0.058 | 46.67% | 1746.75% | -0.0001 | -0.082 |

#### Single Asset Win Rates (%)

| Strategy | AAPL | MSFT | GOOGL | BTC-USD |
|----------|------|------|-------|---------|
| **Keyword** | 52.67 | 60.00 | 54.67 | 49.33 |
| **Hash** | 46.67 | 52.67 | 50.00 | 50.00 |
| **Random** | 48.00 | 49.33 | 48.67 | 44.67 |
| **Momentum** | 46.00 | 48.00 | 54.67 | 48.67 |
| **MeanReversion** | 46.67 | 45.33 | 46.67 | 41.33 |

#### Market Condition Analysis (BTC-USD)

| Strategy | Bull Market | Bear Market | Sideways |
|----------|-------------|-------------|----------|
| **Keyword** | 74.19% | 26.79% | 40.62% |
| **Hash** | 58.06% | 42.86% | 46.88% |
| **Momentum** | 50.00% | 44.64% | 53.12% |
| **MeanReversion** | 43.55% | 42.86% | 46.88% |

### Key Findings

1. **LLM vs Traditional Methods**: No statistically significant advantage (p-value = 0.977 > 0.05)
2. **Market Condition Dependency**: Keyword strategy performs best in bull markets (74.19% win rate)
3. **Ablation Study**: Real News ≈ Shuffled News, indicating limited information content in news
4. **All strategies have Sharpe ratios < 0.2**, suggesting limited practical utility
5. **Error Analysis**: ~28% Easy samples, ~28% Hard samples, ~22% LLM-special, ~21% Hash-special

### Limitations

1. Only 4 assets tested
2. Only simple strategies implemented
3. Transaction costs not optimized for extreme scenarios

### Citation

If you use this code in your research, please cite:

```bibtex
@misc{llmnews2024,
  title={LLM News Analysis Experiment System},
  author={LLM-News Team},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/LLM-news}}
}
```

---

## 中文版本

### 项目简介

本仓库包含评估LLM金融新闻分析策略的代码。我们的研究解决一个根本问题：**大语言模型（LLMs）能否从金融新闻中提取有意义的预测信号用于交易？**

#### 研究问题
- **RQ1**: LLM策略真的优于传统策略吗？
- **RQ2**: 什么策略在什么市场条件下表现最好？
- **RQ3**: 不同策略的优缺点是什么？

### 核心贡献

- **严格统计评估**: T检验、Wilcoxon检验和Bootstrap置信区间，配合Bonferroni校正
- **T+1交易协议**: 严格的时间对齐以避免未来信息泄露
- **6种策略对比**: LLM、Keyword、Hash、Random、Momentum、MeanReversion
- **多资产验证**: 跨股票（AAPL、MSFT、GOOGL）和加密货币（BTC-USD）评估
- **综合分析**: 错误分析、市场条件分析、敏感性分析
- **Publication-ready可视化**: 策略对比图表、累积收益曲线、热力图
- **完整可复现性**: 一键复现，固定随机种子和详细日志

### 项目结构

```
LLM-news/
├── src/                              # 核心模块
│   ├── dataset_adapter.py             # 数据金融化改造
│   ├── strategies.py                  # 统一策略实现
│   ├── strategy_momentum.py           # 动量策略
│   ├── strategy_meanreversion.py      # 均值回归策略
│   ├── llm_model.py                  # LLM集成（带缓存）
│   ├── backtest.py                   # 回测引擎
│   ├── enhanced_stats.py              # 增强统计检验
│   ├── error_analysis.py             # 错误分析模块
│   ├── market_condition_analysis.py  # 市场条件分析
│   ├── sensitivity_analysis.py        # 敏感性分析
│   ├── visualization.py               # 可视化管理器
│   └── experiment_logger.py           # 实验日志
├── experiments/                       # 实验脚本
│   ├── final_experiment.py           # 最终实验（含所有分析）
│   └── sanity_check.py               # Sanity Check
├── scripts/                           # 工具脚本
│   ├── reproduce_all.py              # 一键复现
│   ├── generate_single_asset_results.py  # 单资产结果生成
│   └── check_price_data.py          # 数据质量检查
├── docs/                              # 文档
│   ├── REPRODUCIBILITY.md           # 复现指南
│   ├── DATASET_CARD.md              # 数据说明文档
│   └── EXPERIMENTAL_PROTOCOL.md     # 实验协议
├── data/                              # 数据目录
│   ├── cache/                        # 缓存价格数据
│   └── From_News_to_Forecast/       # NeurIPS 2024数据
├── results/                           # 结果目录
│   ├── tables/                       # 结果表格
│   └── figures/                      # 可视化图表
├── config/                            # 配置文件
│   └── prompts.yaml                  # Prompt模板
├── requirements.txt                   # Python依赖
├── .env.example                      # 环境变量模板
└── README.md                         # 本文件
```

### 快速开始（3步）

```bash
# 步骤1：安装依赖
pip install -r requirements.txt

# 步骤2：配置环境（可选）
cp .env.example .env
# 如果要使用真实LLM API，请编辑.env文件

# 步骤3：运行所有实验
python scripts/reproduce_all.py
```

### 一键复现

```bash
# 一条命令复现所有实验
python scripts/reproduce_all.py

# Windows用户
scripts\reproduce_all.bat
```

**输出**：所有结果保存到 `results/` 目录，包括：
- 策略对比结果
- 统计检验和置信区间
- 错误分析、市场条件分析、敏感性分析
- Publication-ready可视化图表
- 用于可复现性的实验元数据

### 数据说明

| 项目 | 描述 |
|------|------|
| **新闻数据** | 来自Reuters、Bloomberg、CNBC、Yahoo Finance的真实金融新闻 |
| **来源** | From_News_to_Forecast (NeurIPS 2024) |
| **价格数据** | Yahoo Finance（通过yfinance） |
| **资产** | AAPL、MSFT、GOOGL、BTC-USD |
| **时间范围** | 2018-01-01 至 2021-05-11 |
| **样本量** | 每个资产150条新闻 |

### 实验结果

#### 整体表现（BTC-USD）

| 策略 | 收益率 | 夏普比率 | 胜率 | 最大回撤 | 卡玛比率 | 索提诺比率 |
|------|--------|---------|------|---------|---------|-----------|
| **Keyword** | 0.0038 | 0.193 | 52.67% | 321.99% | 0.0012 | 0.368 |
| **Hash** | -0.0005 | -0.026 | 46.67% | 47950.22% | -0.0000 | -0.036 |
| **Random** | -0.0017 | -0.085 | 48.00% | -0.00% | 0.0000 | -0.096 |
| **Momentum** | -0.0019 | -0.093 | 46.00% | 646.15% | -0.0003 | -0.107 |
| **MeanReversion** | -0.0012 | -0.058 | 46.67% | 1746.75% | -0.0001 | -0.082 |

#### 单资产胜率（%）

| 策略 | AAPL | MSFT | GOOGL | BTC-USD |
|------|------|------|-------|---------|
| **Keyword** | 52.67 | 60.00 | 54.67 | 49.33 |
| **Hash** | 46.67 | 52.67 | 50.00 | 50.00 |
| **Random** | 48.00 | 49.33 | 48.67 | 44.67 |
| **Momentum** | 46.00 | 48.00 | 54.67 | 48.67 |
| **MeanReversion** | 46.67 | 45.33 | 46.67 | 41.33 |

#### 市场条件分析（BTC-USD）

| 策略 | 牛市 | 熊市 | 震荡市 |
|------|------|------|--------|
| **Keyword** | 74.19% | 26.79% | 40.62% |
| **Hash** | 58.06% | 42.86% | 46.88% |
| **Momentum** | 50.00% | 44.64% | 53.12% |
| **MeanReversion** | 43.55% | 42.86% | 46.88% |

### 关键发现

1. **LLM vs 传统方法**：无统计学显著优势（p-value = 0.977 > 0.05）
2. **市场条件依赖**：Keyword策略在牛市表现最好（74.19%胜率）
3. **消融研究**：真实新闻 ≈ 打乱新闻，表明新闻信息含量有限
4. **所有策略夏普比率 < 0.2**，表明实际效用有限
5. **错误分析**：约28%Easy样本，约28%Hard样本，约22%LLM专用，约21%Hash专用

### 局限性

1. 只测试了4个资产
2. 只实现了简单策略
3. 交易成本未针对极端情况进行优化

### 引用

如果您在研究中使用本代码，请引用：

```bibtex
@misc{llmnews2024,
  title={LLM新闻分析实验系统},
  author={LLM-News团队},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/LLM-news}}
}
```
