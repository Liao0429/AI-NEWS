import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, List, Any

# 定义不同版本的提示模板
PROMPT_V1 = '''你是一个专业的金融分析师，专注于股票市场分析。
请分析以下新闻，判断它对股票价格的影响：

新闻内容：
{news_text}

请直接返回JSON格式，包含以下字段：
{{
  "sentiment": 1  # 1=看涨，-1=看跌，0=中性
}}
'''

PROMPT_V2 = '''你是一个专业的金融分析师，专注于股票市场分析。
请分析以下新闻，判断它对股票价格的影响：

新闻内容：
{news_text}

请直接返回JSON格式，包含以下字段：
{{
  "sentiment": 1  # 1=看涨，-1=看跌，0=中性，999=无法判断
}}
'''


class LLMModel:
    """LLM模型接口"""
    
    def __init__(self, model_name: str = "kimi-k2", prompt_version: str = "PROMPT_V1", use_cache: bool = True):
        """
        初始化LLM模型
        
        Args:
            model_name: 模型名称
            prompt_version: 提示模板版本
            use_cache: 是否使用缓存
        """
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.use_cache = use_cache
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "parse_failures": 0
        }
        
        # 加载提示模板
        if prompt_version == "PROMPT_V1":
            self.prompt_template = PROMPT_V1
        elif prompt_version == "PROMPT_V2":
            self.prompt_template = PROMPT_V2
        else:
            self.prompt_template = PROMPT_V1
        
        print(f"[OK] LLM client initialized (model: {model_name}, prompt: {prompt_version})")
    
    def _get_cache_key(self, news_text: str) -> str:
        """获取缓存键"""
        content = f"{self.model_name}_{self.prompt_version}_{news_text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[int]:
        """从缓存加载结果"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("sentiment", 0)
            except:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, sentiment: int):
        """保存结果到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"sentiment": sentiment, "timestamp": datetime.now().isoformat()}, f)
        except:
            pass
    
    def _parse_output(self, output: str) -> int:
        """解析LLM输出"""
        try:
            # 尝试直接解析JSON
            data = json.loads(output)
            sentiment = int(data.get("sentiment", 0))
            # 确保返回值是-1, 0, 1
            if sentiment not in [-1, 0, 1]:
                self.stats["parse_failures"] += 1
                return 0
            return sentiment
        except Exception:
            # 解析失败，返回中性
            self.stats["parse_failures"] += 1
            return 0
    
    def generate_signal(self, news_text: str) -> int:
        """
        生成交易信号
        
        Args:
            news_text: 新闻文本
            
        Returns:
            int: 交易信号 (-1=卖出, 0=持有, 1=买入)
        """
        self.stats["total_calls"] += 1
        
        # 检查缓存
        cache_key = self._get_cache_key(news_text)
        if self.use_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.stats["cache_hits"] += 1
                return cached_result
        
        # 生成提示
        prompt = self.prompt_template.format(news_text=news_text)
        
        # 模拟LLM响应（实际应用中应该调用真实的LLM API）
        # 这里使用一个简单的规则来模拟LLM的行为
        sentiment = self._simulate_llm_response(news_text)
        
        # 保存到缓存
        if self.use_cache:
            self._save_to_cache(cache_key, sentiment)
        
        return sentiment
    
    def _simulate_llm_response(self, news_text: str) -> int:
        """
        模拟LLM响应
        
        Args:
            news_text: 新闻文本
            
        Returns:
            int: 情绪得分 (-1=看跌, 0=中性, 1=看涨)
        """
        # 简单的关键词规则模拟
        positive_words = ["上涨", "增长", "盈利", "创新", "成功", "突破", "利好", "上升", "新高", "强劲"]
        negative_words = ["下跌", "亏损", "下滑", "失败", "危机", "利空", "下降", "新低", "疲软", "衰退"]
        
        positive_count = sum(1 for word in positive_words if word in news_text)
        negative_count = sum(1 for word in negative_words if word in news_text)
        
        if positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return -1
        else:
            return 0
    
    def get_stats(self) -> Dict[str, int]:
        """获取模型统计信息"""
        return self.stats
    
    def get_parse_failure_rate(self) -> float:
        """获取解析失败率"""
        total = self.stats.get("total_calls", 1)
        failures = self.stats.get("parse_failures", 0)
        return failures / total


def get_reproducible_llm(prompt_version: str = "PROMPT_V1", model: str = "kimi-k2", use_cache: bool = True) -> LLMModel:
    """
    获取可复现的LLM实例
    
    Args:
        prompt_version: 提示模板版本
        model: 模型名称
        use_cache: 是否使用缓存
        
    Returns:
        LLMModel: LLM模型实例
    """
    return LLMModel(model_name=model, prompt_version=prompt_version, use_cache=use_cache)


def get_prompt_versions() -> List[str]:
    """获取可用的提示模板版本"""
    return ["PROMPT_V1", "PROMPT_V2"]


# 测试代码
if __name__ == "__main__":
    # 测试LLM模型
    llm = get_reproducible_llm(prompt_version="PROMPT_V1", model="kimi-k2", use_cache=True)
    
    # 测试新闻
    test_news = [
        "苹果公司发布新款iPhone，销量超出预期",
        "特斯拉股价下跌5%，因生产数据不及预期",
        "微软宣布重大AI突破，股价上涨",
        "亚马逊财报显示亏损，股价应声下跌"
    ]
    
    print("测试LLM模型:")
    for news in test_news:
        signal = llm.generate_signal(news)
        print(f"新闻: {news}")
        print(f"信号: {signal}")
        print()
    
    # 打印统计信息
    print("统计信息:")
    print(llm.get_stats())
    print(f"解析失败率: {llm.get_parse_failure_rate():.2f}")