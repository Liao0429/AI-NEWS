# Experimental Protocol: LLM新闻分析实验系统

## 摘要

本实验协议详细描述了用于评估LLM在金融新闻分析中有效性的完整实验流程。该协议的目标是确保任何审稿人或研究者可以严格复现所有实验结果。

---

## 1. 数据处理流程

### 1.1 原始数据来源

**新闻数据：**
- 来源：From_News_to_Forecast项目（Wang et al., NeurIPS 2024）
- 数据集：`From_News_to_Forecast/data/raw_news_data/bitcoin_news.json`
- 新闻数量：5,827条真实新闻
- 新闻来源：真实新闻网站（Reuters, Bloomberg, CNBC, Yahoo Finance等）
- 时间范围：2018-01-01 至 2021-05-11

**价格数据：**
- 来源：Yahoo Finance API (yfinance)
- 资产：AAPL, MSFT, GOOGL, BTC-USD
- 频率：每日
- 字段：Open, High, Low, Close, Volume
- 时间范围：2017-12-02 至 2021-06-10

### 1.2 数据清洗步骤

**新闻数据清洗：**

1. **日期格式标准化**
   - 输入格式：`YYYY-MM-DD HH:MM:SS`
   - 输出格式：`YYYY-MM-DD`

2. **去除无效数据**
   - 删除缺少日期的新闻
   - 删除缺少标题的新闻
   - 删除日期不在价格数据范围内的新闻

3. **文本标准化**
   - 保留原始新闻文本
   - 同时使用标题和完整文章

**价格数据清洗：**

1. **缺失值处理**
   - 使用`pd.to_numeric()`转换为数值类型
   - 删除缺失值

2. **异常值处理**
   - 移除价格变化超过5个标准差的极端值

3. **MultiIndex处理**
   - 扁平化yfinance返回的MultiIndex列

### 1.3 时间对齐规则（T+1严格执行）

**核心原则：** 严禁数据泄露（no look-ahead bias）

**时间定义：**

1. **news_time**：新闻发布时间
   - 定义：新闻发布日期的17:00:00（收盘后）

2. **prediction_time**：LLM预测时间
   - 定义：news_time + 1小时

3. **trade_time**：交易时间
   - 定义：下一个交易日的09:30:00（开盘时）
   - 规则：跳过周末和非交易日

4. **future_price_time**：未来价格时间
   - 定义：trade_time的下一个交易日的16:00:00（收盘时）

**严格时间顺序验证：**
```
news_time < prediction_time < trade_time < future_price_time
```

---

## 2. 交易规则

### 2.1 信号生成

**信号空间：**
- `signal ∈ {+1, -1}`
- +1：做多（long）
- -1：做空（short）

### 2.2 策略列表

**测试的6种策略：**

1. **LLM策略**
   - 方法：使用LLM API或关键词方法
   - 信号：基于新闻文本语义分析

2. **Keyword策略**
   - 方法：正面/负面关键词匹配
   - 关键词列表：
     - 正面词：rise, gain, positive, strong, beat, up, growth, increase, success, bull, soar, surge, rally, jump
     - 负面词：fall, drop, negative, weak, miss, down, decline, decrease, failure, bear, crash, plunge, slump, tumble

3. **Hash策略（Rule策略）**
   - 方法：确定性哈希
   - 实现：使用MD5哈希生成信号

4. **Random策略**
   - 方法：完全随机
   - 信号：50%概率+1，50%概率-1

5. **Momentum策略**
   - 方法：动量策略
   - 参数：时间窗口（默认5天）
   - 逻辑：价格上涨则做多，下跌则做空

6. **MeanReversion策略**
   - 方法：均值回归策略
   - 参数：时间窗口（默认20天）
   - 逻辑：价格高于均值则做空，低于均值则做多

### 2.3 收益计算公式

**单条新闻收益：**
```
future_return = (future_price - trade_price) / trade_price
strategy_return = signal * future_return
```

### 2.4 交易成本

- 佣金：0.001（0.1%）
- 滑点：0.001（0.1%）
- 计算公式：`gross_return = signal * future_return; net_return = gross_return - commission - slippage`

---

## 3. 实验设置

### 3.1 回测设置

**回测参数：**
- 样本数量：150条新闻/资产
- 随机种子：42（固定）
- 资产数量：4个（AAPL, MSFT, GOOGL, BTC-USD）
- 时间范围：2018-01-01 至 2021-05-11

**资产配置：**
| 资产 | 说明 | 数据天数 |
|------|------|---------|
| AAPL | Apple Inc. | 881天 |
| MSFT | Microsoft Corp. | 881天 |
| GOOGL | Alphabet Inc. | 881天 |
| BTC-USD | Bitcoin/USD | 1288天 |

### 3.2 多次运行设置

- 运行次数：1次（单次运行，使用固定种子）
- 统计检验：配对t检验或Wilcoxon检验

---

## 4. 统计方法

### 4.1 正态性检验

**Shapiro-Wilk检验：**
- 目的：检验数据是否符合正态分布
- 实现：`scipy.stats.shapiro()`
- 阈值：样本量 >= 20

**结果处理：**
- p-value > 0.05：数据符合正态分布，使用配对t检验
- p-value ≤ 0.05：数据不符合正态分布，使用Wilcoxon检验

### 4.2 配对t检验

**目的：** 比较两个策略的平均收益是否有显著差异

**公式：**
```
t = (mean(differences)) / (std(differences) / sqrt(n))
```

**实现：**
- 库：`scipy.stats.ttest_rel`
- 显著性水平：α = 0.05

### 4.3 Wilcoxon检验

**目的：** 非参数检验，用于非正态数据

**实现：**
- 库：`scipy.stats.wilcoxon`

### 4.4 Bootstrap置信区间

**参数：**
- Bootstrap次数：1000次
- 方法：有放回重采样

**算法：**
```
for b in 1 to 1000:
    resampled_returns = sample_with_replacement(original_returns)
    bootstrap_mean[b] = mean(resampled_returns)

CI_lower = percentile(bootstrap_mean, 2.5)
CI_upper = percentile(bootstrap_mean, 97.5)
```

### 4.5 Bonferroni校正

**目的：** 处理多重假设检验

**公式：**
```
校正后的α = α / n
其中 n = 检验次数
```

---

## 5. 评价指标

### 5.1 收益（Return）

**公式：**
```
mean_return = (1/N) * Σ(strategy_return_i)
```

### 5.2 夏普比率（Sharpe Ratio）

**公式：**
```
sharpe_ratio = mean_return / std_return
```

### 5.3 胜率（Win Rate）

**公式：**
```
win_rate = (number of positive returns) / (total number of returns) * 100%
```

### 5.4 最大回撤（Max Drawdown）

**公式：**
```
running_max = cummax(returns)
drawdown = (returns - running_max) / running_max
max_drawdown = min(drawdown)
```

### 5.5 卡玛比率（Calmar Ratio）

**公式：**
```
calmar_ratio = mean_return / abs(max_drawdown)
```

### 5.6 索提诺比率（Sortino Ratio）

**公式：**
```
downside_returns = returns[returns < 0]
downside_std = std(downside_returns)
sortino_ratio = mean_return / downside_std
```

---

## 6. 深入分析

### 6.1 错误分析

**样本分类：**
1. **Easy样本**：两个策略都正确
2. **Hard样本**：两个策略都错误
3. **LLM-special样本**：只有LLM正确
4. **Hash-special样本**：只有Hash正确

### 6.2 市场条件分析

**市场状态定义：**
- **牛市（Bull）**：最近20天价格上涨 > 2%
- **熊市（Bear）**：最近20天价格下跌 > 2%
- **震荡市（Sideways）**：其他情况

**分析内容：**
- 各策略在不同市场状态下的胜率
- 热力图可视化

### 6.3 敏感性分析

**测试参数：**
- Momentum策略：时间窗口（1, 3, 5, 10, 15, 20, 30天）
- MeanReversion策略：时间窗口（5, 10, 15, 20, 30, 40, 50天）

---

## 7. 实验流程

### 7.1 数据准备

1. 加载From_News_to_Forecast新闻数据
2. 下载Yahoo Finance价格数据
3. 数据清洗和质量控制
4. T+1时间对齐

### 7.2 策略运行

对每个资产运行6种策略：
1. Keyword策略
2. Hash策略
3. Random策略
4. Momentum策略
5. MeanReversion策略
6. LLM策略（如有API）

### 7.3 统计分析

1. 计算7个评价指标
2. 配对统计检验
3. Bootstrap置信区间
4. Bonferroni校正

### 7.4 深入分析

1. 错误分析
2. 市场条件分析
3. 敏感性分析

### 7.5 可视化

生成9种图表：
1. 策略对比柱状图
2. 收益分布箱线图
3. 多指标雷达图
4. 累积收益曲线
5. 月度收益热力图
6. 策略表现趋势图
7. 错误分析饼图
8. 市场条件热力图
9. 敏感性分析曲线

---

## 8. 实验结果

### 8.1 单资产结果（BTC-USD）

| 策略 | 收益率 | 夏普比率 | 胜率 | 最大回撤 | 卡玛比率 | 索提诺比率 |
|------|--------|---------|------|---------|---------|-----------|
| **Keyword** | 0.0038 | 0.193 | 52.67% | 321.99% | 0.0012 | 0.368 |
| **Hash** | -0.0005 | -0.026 | 46.67% | 47950.22% | -0.0000 | -0.036 |
| **Random** | -0.0017 | -0.085 | 48.00% | -0.00% | 0.0000 | -0.096 |
| **Momentum** | -0.0019 | -0.093 | 46.00% | 646.15% | -0.0003 | -0.107 |
| **MeanReversion** | -0.0012 | -0.058 | 46.67% | 1746.75% | -0.0001 | -0.082 |

### 8.2 单资产胜率对比

| 策略 | AAPL | MSFT | GOOGL | BTC-USD |
|------|------|------|-------|---------|
| **Keyword** | 52.67% | 60.00% | 54.67% | 49.33% |
| **Hash** | 46.67% | 52.67% | 50.00% | 50.00% |
| **Random** | 48.00% | 49.33% | 48.67% | 44.67% |
| **Momentum** | 46.00% | 48.00% | 54.67% | 48.67% |
| **MeanReversion** | 46.67% | 45.33% | 46.67% | 41.33% |

### 8.3 市场条件分析（BTC-USD）

| 策略 | 牛市 | 熊市 | 震荡市 |
|------|------|------|--------|
| **Keyword** | 74.19% | 26.79% | 40.62% |
| **Hash** | 58.06% | 42.86% | 46.88% |
| **Momentum** | 50.00% | 44.64% | 53.12% |
| **MeanReversion** | 43.55% | 42.86% | 46.88% |

### 8.4 关键发现

1. **LLM vs 传统方法**：无统计学显著优势（p-value = 0.977 > 0.05）
2. **Keyword在牛市表现最好**：74.19%胜率
3. **所有策略夏普比率 < 0.2**：实际效用有限
4. **错误分析**：约28%Easy，约28%Hard，约22%LLM-special，约21%Hash-special

---

## 9. 可复现性保证

### 9.1 代码和数据

**代码：**
- 完整开源：提供所有源代码
- 模块化设计：清晰的模块划分
- 详细注释：关键步骤有注释

**数据：**
- 真实数据：From_News_to_Forecast（NeurIPS 2024）
- 价格数据：可通过yfinance重新下载

### 9.2 随机种子

**固定的种子：**
- 主随机种子：42
- 所有策略使用相同的随机种子

### 9.3 LLM可复现性

**固定参数：**
- temperature = 0
- top_p = 1
- max_tokens = 200

**缓存机制：**
- 输入：news_text + model_name + prompt_version
- 输出：prediction + confidence
- 缓存文件：`data/cache/llm_cache.json`

### 9.4 依赖版本

**关键依赖：**
- Python: 3.8+
- pandas: 1.5.0+
- numpy: 1.23.0+
- scipy: 1.9.0+
- yfinance: 0.2.0+
- matplotlib: 3.5.0+

---

## 10. 结果解释规范

### 10.1 统计显著性

**规则：**
- 所有统计检验使用α = 0.05
- Bonferroni校正后α = 0.05/6 ≈ 0.008
- p-value > 0.05：报告"无显著差异"
- p-value ≤ 0.05：报告"有显著差异"

### 10.2 结果报告格式

**表格格式：**
```
| Strategy | Return | Sharpe | Win Rate | p-value |
|----------|--------|--------|----------|---------|
| Keyword  | 0.0038 | 0.193  | 52.67%   | -       |
| Hash     | -0.0005| -0.026 | 46.67%   | 0.977   |
```

**精度要求：**
- Return：4位小数
- Sharpe：4位小数
- Win Rate：2位小数（百分比）
- p-value：3位小数

---

## 参考文献

Wang, X., Feng, M., Qiu, J., Gu, J., & Zhao, J. (2024). From News to Forecast: Integrating Event Analysis in LLM-based Time Series Forecasting with Reflection. NeurIPS 2024.

---

**文档版本：** 2.0
**最后更新：** 2026-04-06
**维护者：** LLM新闻分析实验团队
