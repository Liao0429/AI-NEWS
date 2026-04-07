# Data Card: LLM新闻分析实验数据集

## 摘要

本数据说明文档详细描述了用于评估LLM在金融新闻分析中有效性的数据集。该文档的目标是确保数据使用合法、透明、可审查。

---

## 1. 数据来源

### 1.1 新闻数据

**主要数据源：From_News_to_Forecast项目**
- **论文**：From News to Forecast: Integrating Event Analysis in LLM-based Time Series Forecasting with Reflection
- **会议**：NeurIPS 2024
- **作者**：Wang, X., Feng, M., Qiu, J., Gu, J., & Zhao, J.
- **项目链接**：https://github.com/ameliawong1996/From_News_to_Forecast
- **数据集位置**：`From_News_to_Forecast/data/raw_news_data/bitcoin_news.json`
- **是否公开**：✅ 公开数据集（随论文发布）

**子数据源：**
- GDELT Project: 全球新闻数据库
- Yahoo Finance: 金融新闻
- News AU: 澳大利亚新闻

### 1.2 价格数据

**数据源：Yahoo Finance API**
- **API**：yfinance (Python库)
- **是否公开**：✅ 公开API
- **获取方式**：通过yfinance库下载

---

## 2. 数据结构

### 2.1 原始新闻数据结构

**文件**：`From_News_to_Forecast/data/raw_news_data/bitcoin_news.json`

**格式**：JSON数组

**字段说明：**

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `title` | string | 新闻标题 | "Bitcoin Rises 5% on Strong Demand" |
| `full_article` | string | 完整文章 | "[完整新闻文本...]" |
| `pub_time` | string | 发布时间 | "2020-05-01 14:30:00" |
| `source` | string | 新闻来源 | "Reuters" |

### 2.2 处理后的数据结构

**文件**：内存中处理（不保存到磁盘）

**字段说明：**

| 字段 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `news_time` | datetime | 新闻发布时间 | pub_time标准化（17:00:00） |
| `news_date` | date | 新闻发布日期 | pub_time日期部分 |
| `news_text` | string | 新闻文本 | title + full_article |
| `source` | string | 新闻来源 | 原始source字段 |
| `prediction_time` | datetime | LLM预测时间 | news_time + 1小时 |
| `trade_time` | datetime | 交易时间 | 下一个交易日09:30:00 |
| `trade_date` | date | 交易日期 | trade_time日期部分 |
| `future_price_time` | datetime | 未来价格时间 | trade_time的下一个交易日16:00:00 |
| `future_price_date` | date | 未来价格日期 | future_price_time日期部分 |
| `open` | float | 开盘价 | yfinance下载 |
| `high` | float | 最高价 | yfinance下载 |
| `low` | float | 最低价 | yfinance下载 |
| `close` | float | 收盘价 | yfinance下载 |
| `volume` | int | 成交量 | yfinance下载 |

### 2.3 严格时间顺序验证

**核心原则：** 严禁数据泄露（no look-ahead bias）

**时间顺序：**
```
news_time < prediction_time < trade_time < future_price_time
```

**验证：**
- 所有数据点都验证此时间顺序
- 违反时间顺序的数据点被丢弃

---

## 3. 时间范围

### 3.1 新闻数据时间范围

- **开始时间**：2018-01-01
- **结束时间**：2021-05-11
- **总天数**：1,226天
- **有效新闻数量**：5,827条

### 3.2 价格数据时间范围

- **开始时间**：2017-12-01（新闻日期前30天）
- **结束时间**：2021-06-10（新闻日期后30天）
- **频率**：每日
- **资产**：BTC-USD

### 3.3 实验使用的数据子集

- **新闻数量**：150条（均匀采样）
- **时间范围**：2018-01-01 至 2021-05-11
- **采样方法**：均匀采样，覆盖整个时间范围

---

## 4. 数据限制

### 4.1 缺失值

**新闻数据：**
- 缺失日期的新闻：已删除
- 缺失标题的新闻：已删除
- 日期不在价格范围内的新闻：已删除
- **最终缺失率**：< 1%

**价格数据：**
- 缺失值处理：前向填充（forward fill）
- **最终缺失率**：< 0.1%

### 4.2 数据偏差

**潜在偏差：**

1. **来源偏差**
   - 主要来源：Reuters, Bloomberg, CNBC, Yahoo Finance
   - 偏向英语新闻
   - 偏向主流金融媒体

2. **时间偏差**
   - 时间范围：2018-2021
   - 包含：比特币牛市（2020-2021）
   - 不包含：2022年熊市
   - 不包含：2017年牛市

3. **资产偏差**
   - 单一资产：BTC-USD
   - 未测试：股票（AAPL, MSFT等）
   - 未测试：其他加密货币

4. **采样偏差**
   - 实验使用150条子集
   - 均匀采样，但样本量较小

### 4.3 数据质量

**数据质量评估：**

| 方面 | 评分 | 说明 |
|------|------|------|
| 真实性 | ⭐⭐⭐⭐⭐ | 来自真实新闻网站 |
| 完整性 | ⭐⭐⭐⭐ | 少量缺失值已处理 |
| 时效性 | ⭐⭐⭐⭐ | 2018-2021（较新但非最新） |
| 多样性 | ⭐⭐⭐ | 单一资产，英语新闻 |

---

## 5. 版权说明

### 5.1 新闻数据

**From_News_to_Forecast项目：**
- **版权**：属于原作者
- **许可**：请参考原项目的LICENSE文件
- **使用方式**：随论文发布，用于研究目的
- **full_text**：✅ 提供完整新闻文本（包含在原数据集中）

### 5.2 价格数据

**Yahoo Finance API：**
- **版权**：Yahoo Finance
- **使用方式**：通过yfinance库获取
- **限制**：仅供个人/研究使用
- **重新分发**：不建议重新分发价格数据

### 5.3 本项目数据

**本项目中的数据：**
- **来源**：From_News_to_Forecast + Yahoo Finance
- **使用目的**：学术研究
- **重新分发**：建议引用原论文，不直接重新分发数据
- **引用要求**：见下方引用部分

---

## 6. 使用方式

### 6.1 数据加载

**方法1：从From_News_to_Forecast加载（推荐）**

```python
from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data

# 加载并预处理数据
aligned_df = load_and_prepare_from_news_to_forecast_data()

# 查看数据
print(f"Loaded {len(aligned_df)} data points")
print(aligned_df[['news_time', 'trade_time', 'future_price_time']].head())
```

**方法2：手动加载原始数据**

```python
import json
import pandas as pd

# 加载新闻数据
with open('From_News_to_Forecast/data/raw_news_data/bitcoin_news.json', 'r') as f:
    news_data = json.load(f)

# 转换为DataFrame
news_df = pd.DataFrame(news_data)
print(f"Loaded {len(news_df)} news articles")

# 加载价格数据（使用yfinance）
import yfinance as yf

price_data = yf.download('BTC-USD', start='2018-01-01', end='2021-05-31')
print(f"Loaded {len(price_data)} price points")
```

### 6.2 数据预处理

**时间对齐（T+1严格执行）：**

```python
from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data

# 自动完成所有预处理
aligned_df = load_and_prepare_from_news_to_forecast_data()

# 验证时间顺序
for _, row in aligned_df.iterrows():
    assert row['news_time'] < row['prediction_time']
    assert row['prediction_time'] < row['trade_time']
    assert row['trade_time'] < row['future_price_time']

print("✓ Time order verified")
```

### 6.3 数据使用示例

**回测示例：**

```python
import pandas as pd
import numpy as np

# 加载数据
aligned_df = load_and_prepare_from_news_to_forecast_data()

# 计算收益
returns = []
for _, row in aligned_df.iterrows():
    future_return = (row['future_price'] - row['trade_price']) / row['trade_price']
    signal = 1  # 示例：总是做多
    strategy_return = signal * future_return
    returns.append(strategy_return)

# 计算指标
mean_return = np.mean(returns)
sharpe = mean_return / np.std(returns) if np.std(returns) > 0 else 0
win_rate = np.mean([1 if r > 0 else 0 for r in returns]) * 100

print(f"Return: {mean_return:.6f}")
print(f"Sharpe: {sharpe:.4f}")
print(f"Win Rate: {win_rate:.2f}%")
```

### 6.4 数据字段访问

**字段访问示例：**

```python
# 访问新闻文本
news_text = aligned_df.iloc[0]['news_text']
print(f"News text: {news_text[:100]}...")

# 访问时间信息
news_time = aligned_df.iloc[0]['news_time']
trade_time = aligned_df.iloc[0]['trade_time']
future_price_time = aligned_df.iloc[0]['future_price_time']

print(f"News time: {news_time}")
print(f"Trade time: {trade_time}")
print(f"Future price time: {future_price_time}")

# 验证时间差
time_diff_1 = (trade_time - news_time).total_seconds() / 3600
time_diff_2 = (future_price_time - trade_time).total_seconds() / 3600

print(f"Time from news to trade: {time_diff_1:.1f} hours")
print(f"Time from trade to future price: {time_diff_2:.1f} hours")
```

---

## 7. 引用

如果您使用了本数据集，请引用：

```bibtex
@inproceedings{wang2024news,
  title={From News to Forecast: Integrating Event Analysis in LLM-based Time Series Forecasting with Reflection},
  author={Wang, Xinlei and Feng, Maike and Qiu, Jing and Gu, Jinjin and Zhao, Junhua},
  booktitle={NeurIPS},
  year={2024}
}
```

---

## 8. 联系方式

如有数据相关问题，请：
1. 参考原论文：From_News_to_Forecast (NeurIPS 2024)
2. 访问原项目：https://github.com/ameliawong1996/From_News_to_Forecast
3. 提交Issue到本项目

---

**文档版本：** 1.0  
**最后更新：** 2026-04-06  
**维护者：** LLM新闻分析实验团队
