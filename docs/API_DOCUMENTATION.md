# API Documentation / API文档

---

## English Version

### Core Modules

#### 1. Dataset Adapter (`src/dataset_adapter.py`)

```python
class DataProcessor:
    """Data processor for financial news and price data"""

    def load_from_news_to_forecast_data(self, ticker: str, sample_size: int = 150) -> pd.DataFrame:
        """
        Load and prepare data from From_News_to_Forecast dataset

        Args:
            ticker: Asset ticker (AAPL, MSFT, GOOGL, BTC-USD)
            sample_size: Number of samples to use

        Returns:
            DataFrame with news and aligned price data
        """

    def load_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load price data from Yahoo Finance

        Args:
            ticker: Asset ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
```

#### 2. Strategies (`src/strategies.py`)

```python
class BaseStrategy:
    """Base class for all trading strategies"""

    def __init__(self, name: str, random_seed: int = 42):
        """
        Initialize strategy

        Args:
            name: Strategy name
            random_seed: Random seed for reproducibility
        """

    def generate_signal(self, input_data: str) -> int:
        """
        Generate trading signal

        Args:
            input_data: News text or price data

        Returns:
            Signal: +1 (long), -1 (short), 0 (hold)
        """

    def run(self, df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """
        Run strategy on dataframe

        Returns:
            Tuple of (signals, returns)
        """


class KeywordStrategy(BaseStrategy):
    """Keyword-based sentiment analysis strategy"""
    # Uses positive/negative word matching


class HashStrategy(BaseStrategy):
    """Deterministic hash-based strategy (Rule strategy)"""
    # Uses MD5 hash to generate deterministic signals


class RandomStrategy(BaseStrategy):
    """Random baseline strategy"""
    # Generates random ±1 signals


class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy"""

    def __init__(self, time_window: int = 5):
        """
        Args:
            time_window: Number of days for momentum calculation
        """


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""

    def __init__(self, time_window: int = 20):
        """
        Args:
            time_window: Number of days for moving average
        """
```

#### 3. LLM Model (`src/llm_model.py`)

```python
def get_reproducible_llm(prompt_version: str = "PROMPT_V1",
                         model: str = "kimi-k2") -> ReproducibleLLM:
    """
    Get reproducible LLM instance

    Args:
        prompt_version: Version of prompt to use
        model: Model name (kimi-k2, gpt-4, etc.)

    Returns:
        ReproducibleLLM instance
    """


class ReproducibleLLM:
    """LLM with reproducibility guarantees"""

    def predict(self, news_text: str) -> Dict[str, Any]:
        """
        Predict sentiment from news text

        Args:
            news_text: Input news text

        Returns:
            Dict with keys: signal (+1/-1), prediction, confidence
        """

    def batch_predict(self, news_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch prediction

        Args:
            news_texts: List of news texts

        Returns:
            List of prediction dicts
        """
```

#### 4. Statistical Analysis (`src/enhanced_stats.py`)

```python
def compare_strategies(strategy1_returns: List[float],
                       strategy2_returns: List[float],
                       metric: str = 'mean') -> Dict[str, Any]:
    """
    Compare two strategies using statistical tests

    Args:
        strategy1_returns: Returns of strategy 1
        strategy2_returns: Returns of strategy 2
        metric: Metric to compare (mean, sharpe, win_rate, etc.)

    Returns:
        Dict with test results
    """


def generate_statistical_report(strategy_results: Dict[str, Dict[str, float]]) -> str:
    """
    Generate comprehensive statistical report

    Returns:
        Formatted report string
    """
```

#### 5. Error Analysis (`src/error_analysis.py`)

```python
class ErrorAnalyzer:
    """Analyze prediction errors"""

    def __init__(self, aligned_df: pd.DataFrame,
                 keyword_signals: List[int],
                 hash_signals: List[int]):
        """
        Initialize error analyzer

        Args:
            aligned_df: Aligned dataframe with news and prices
            keyword_signals: Keyword strategy signals
            hash_signals: Hash strategy signals
        """

    def classify_samples(self) -> Dict[str, List[int]]:
        """
        Classify samples into Easy/Hard/LLM-special/Rule-special

        Returns:
            Dict with classification results
        """

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get error analysis statistics

        Returns:
            Dict with statistics
        """
```

#### 6. Market Condition Analysis (`src/market_condition_analysis.py`)

```python
class MarketConditionAnalyzer:
    """Analyze strategy performance under different market conditions"""

    def __init__(self, aligned_df: pd.DataFrame):
        """
        Initialize market condition analyzer
        """

    def get_strategy_performance(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get strategy performance by market condition

        Returns:
            Dict: {strategy: {market_condition: {metric: value}}}
        """
```

#### 7. Sensitivity Analysis (`src/sensitivity_analysis.py`)

```python
class SensitivityAnalyzer:
    """Analyze strategy sensitivity to parameters"""

    def __init__(self, aligned_df: pd.DataFrame):
        """
        Initialize sensitivity analyzer
        """

    def analyze_momentum_window(self) -> pd.DataFrame:
        """
        Analyze momentum strategy sensitivity to time window

        Returns:
            DataFrame with sensitivity results
        """

    def analyze_mean_reversion_window(self) -> pd.DataFrame:
        """
        Analyze mean reversion strategy sensitivity to time window

        Returns:
            DataFrame with sensitivity results
        """
```

#### 8. Visualization (`src/visualization.py`)

```python
class VisualizationManager:
    """Generate publication-ready visualizations"""

    def __init__(self, output_dir: str = 'results/figures'):
        """
        Initialize visualization manager
        """

    def plot_strategy_comparison(self, strategy_results: Dict) -> str:
        """Generate strategy comparison bar chart"""

    def plot_return_distribution(self, strategy_returns: Dict) -> str:
        """Generate return distribution box plot"""

    def plot_radar_chart(self, strategy_results: Dict) -> str:
        """Generate multi-metric radar chart"""

    def plot_cumulative_returns(self, strategy_returns: Dict) -> str:
        """Generate cumulative returns curve"""

    def plot_market_condition_heatmap(self, market_performance: Dict) -> str:
        """Generate market condition heatmap"""

    def plot_sensitivity_curve(self, sensitivity_data: Dict) -> str:
        """Generate sensitivity analysis curve"""

    def generate_all_visualizations(self, ...) -> List[str]:
        """Generate all visualizations at once"""
```

---

## 中文版本

### 核心模块

#### 1. 数据适配器 (`src/dataset_adapter.py`)

```python
class DataProcessor:
    """金融新闻和价格数据处理器"""

    def load_from_news_to_forecast_data(self, ticker: str, sample_size: int = 150) -> pd.DataFrame:
        """
        从From_News_to_Forecast数据集加载和准备数据

        参数:
            ticker: 资产代码 (AAPL, MSFT, GOOGL, BTC-USD)
            sample_size: 使用的样本数量

        返回:
            包含新闻和对齐价格数据的DataFrame
        """

    def load_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从Yahoo Finance加载价格数据

        参数:
            ticker: 资产代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        返回:
            包含OHLCV数据的DataFrame
        """
```

#### 2. 策略 (`src/strategies.py`)

```python
class BaseStrategy:
    """所有交易策略的基类"""

    def __init__(self, name: str, random_seed: int = 42):
        """
        初始化策略

        参数:
            name: 策略名称
            random_seed: 随机种子，用于可复现性
        """

    def generate_signal(self, input_data: str) -> int:
        """
        生成交易信号

        参数:
            input_data: 新闻文本或价格数据

        返回:
            信号: +1 (做多), -1 (做空), 0 (持有)
        """

    def run(self, df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """
        在DataFrame上运行策略

        返回:
            (信号列表, 收益率列表)
        """


class KeywordStrategy(BaseStrategy):
    """基于关键词的情绪分析策略"""
    # 使用正面/负面词汇匹配


class HashStrategy(BaseStrategy):
    """基于确定性哈希的策略（规则策略）"""
    # 使用MD5哈希生成确定性信号


class RandomStrategy(BaseStrategy):
    """随机基线策略"""
    # 生成随机±1信号


class MomentumStrategy(BaseStrategy):
    """动量交易策略"""

    def __init__(self, time_window: int = 5):
        """
        参数:
            time_window: 动量计算的天数
        """


class MeanReversionStrategy(BaseStrategy):
    """均值回归交易策略"""

    def __init__(self, time_window: int = 20):
        """
        参数:
            time_window: 移动平均的天数
        """
```

#### 3. LLM模型 (`src/llm_model.py`)

```python
def get_reproducible_llm(prompt_version: str = "PROMPT_V1",
                         model: str = "kimi-k2") -> ReproducibleLLM:
    """
    获取可复现的LLM实例

    参数:
        prompt_version: 使用的提示版本
        model: 模型名称 (kimi-k2, gpt-4, 等)

    返回:
        ReproducibleLLM实例
    """


class ReproducibleLLM:
    """具有可复现性保证的LLM"""

    def predict(self, news_text: str) -> Dict[str, Any]:
        """
        从新闻文本预测情绪

        参数:
            news_text: 输入新闻文本

        返回:
            包含signal、prediction、confidence的字典
        """

    def batch_predict(self, news_texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测

        参数:
            news_texts: 新闻文本列表

        返回:
            预测字典列表
        """
```

#### 4. 统计分析 (`src/enhanced_stats.py`)

```python
def compare_strategies(strategy1_returns: List[float],
                       strategy2_returns: List[float],
                       metric: str = 'mean') -> Dict[str, Any]:
    """
    使用统计检验比较两种策略

    参数:
        strategy1_returns: 策略1的收益率
        strategy2_returns: 策略2的收益率
        metric: 比较的指标 (mean, sharpe, win_rate, 等)

    返回:
            包含检验结果的字典
    """


def generate_statistical_report(strategy_results: Dict[str, Dict[str, float]]) -> str:
    """
    生成综合统计报告

    返回:
            格式化的报告字符串
    """
```

#### 5. 错误分析 (`src/error_analysis.py`)

```python
class ErrorAnalyzer:
    """分析预测错误"""

    def __init__(self, aligned_df: pd.DataFrame,
                 keyword_signals: List[int],
                 hash_signals: List[int]):
        """
        初始化错误分析器

        参数:
            aligned_df: 包含新闻和价格的对齐DataFrame
            keyword_signals: Keyword策略信号
            hash_signals: Hash策略信号
        """

    def classify_samples(self) -> Dict[str, List[int]]:
        """
        将样本分类为Easy/Hard/LLM-special/Rule-special

        返回:
            包含分类结果的字典
        """

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取错误分析统计

        返回:
            包含统计数据的字典
        """
```

#### 6. 市场条件分析 (`src/market_condition_analysis.py`)

```python
class MarketConditionAnalyzer:
    """分析不同市场条件下策略表现"""

    def __init__(self, aligned_df: pd.DataFrame):
        """
        初始化市场条件分析器
        """

    def get_strategy_performance(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        获取按市场条件分类的策略表现

        返回:
            字典: {策略: {市场条件: {指标: 值}}}
        """
```

#### 7. 敏感性分析 (`src/sensitivity_analysis.py`)

```python
class SensitivityAnalyzer:
    """分析策略对参数的敏感性"""

    def __init__(self, aligned_df: pd.DataFrame):
        """
        初始化敏感性分析器
        """

    def analyze_momentum_window(self) -> pd.DataFrame:
        """
        分析动量策略对时间窗口的敏感性

        返回:
            包含敏感性结果的DataFrame
        """

    def analyze_mean_reversion_window(self) -> pd.DataFrame:
        """
        分析均值回归策略对时间窗口的敏感性

        返回:
            包含敏感性结果的DataFrame
        """
```

#### 8. 可视化 (`src/visualization.py`)

```python
class VisualizationManager:
    """生成publication-ready可视化"""

    def __init__(self, output_dir: str = 'results/figures'):
        """
        初始化可视化管理器
        """

    def plot_strategy_comparison(self, strategy_results: Dict) -> str:
        """生成策略对比柱状图"""

    def plot_return_distribution(self, strategy_returns: Dict) -> str:
        """生成收益率分布箱线图"""

    def plot_radar_chart(self, strategy_results: Dict) -> str:
        """生成多指标雷达图"""

    def plot_cumulative_returns(self, strategy_returns: Dict) -> str:
        """生成累积收益曲线"""

    def plot_market_condition_heatmap(self, market_performance: Dict) -> str:
        """生成市场条件热力图"""

    def plot_sensitivity_curve(self, sensitivity_data: Dict) -> str:
        """生成敏感性分析曲线"""

    def generate_all_visualizations(self, ...) -> List[str]:
        """一次生成所有可视化"""
```
