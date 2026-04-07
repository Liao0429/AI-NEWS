import sys
sys.path.append('.')

import os
import json
import time
import random
import hashlib
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 尝试导入OpenAI库
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI库未安装，将使用fallback方法")

# 尝试导入tenacity库
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    print("⚠️  tenacity库未安装，将使用简单重试")


class LLMFinancialPredictorOptimized:
    """
    优化版LLM预测模块
    
    包含：
    1. 指数退避重试机制 (Exponential Backoff)
    2. 异步并发控制 (Async with Semaphore)
    3. 混合架构：LLM + 关键词预过滤 (Hybrid Filter)
    4. 缓存机制
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model: str = "gpt-4", fallback_to_keyword: bool = True,
                 use_cache: bool = True, cache_file: str = "data/llm_cache_optimized.json",
                 use_hybrid_filter: bool = True, hybrid_threshold: int = 0,
                 max_concurrent: int = 3):
        """
        初始化优化版LLM预测器
        
        Args:
            api_key: LLM API key
            base_url: API基础URL
            model: 模型名称
            fallback_to_keyword: API失败时是否fallback到keyword方法
            use_cache: 是否使用缓存机制
            cache_file: 缓存文件路径
            use_hybrid_filter: 是否使用混合架构（关键词预过滤）
            hybrid_threshold: 混合阈值（正面词-负面词的绝对值 <= 此值时才调用LLM）
            max_concurrent: 最大并发数
        """
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("KIMI_BASE_URL", "https://apis.iflow.cn/v1")
        self.model = model
        self.fallback_to_keyword = fallback_to_keyword
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.use_hybrid_filter = use_hybrid_filter
        self.hybrid_threshold = hybrid_threshold
        self.max_concurrent = max_concurrent
        
        # 缓存相关
        self.cache = {}
        self._load_cache()
        
        # 统计信息
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "hybrid_filter_hits": 0,
            "api_calls": 0,
            "fallback_calls": 0
        }
        
        # 初始化日志
        self.logs = []
        self.log_file = "data/llm_raw_outputs_optimized.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        # 初始化OpenAI客户端（如果可用）
        self.client = None
        self.async_client = None
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self.async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                print(f"✓ LLM客户端初始化成功 (model: {self.model})")
                if self.use_cache:
                    print(f"✓ 缓存机制已启用 (缓存文件: {self.cache_file})")
                if self.use_hybrid_filter:
                    print(f"✓ 混合架构已启用 (阈值: {self.hybrid_threshold})")
                print(f"✓ 最大并发数: {self.max_concurrent}")
            except Exception as e:
                print(f"⚠️  LLM客户端初始化失败: {e}")
                self.client = None
                self.async_client = None
        else:
            print("⚠️  LLM客户端未初始化，将使用fallback方法")
    
    def _get_cache_key(self, news_text: str) -> str:
        """生成缓存key"""
        text_hash = hashlib.md5(news_text.encode('utf-8')).hexdigest()
        return f"{self.model}:{text_hash}"
    
    def _load_cache(self) -> None:
        """加载缓存"""
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"✓ 缓存已加载 ({len(self.cache)} 条)")
            except Exception as e:
                print(f"⚠️  缓存加载失败: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """保存缓存"""
        if self.use_cache:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️  缓存保存失败: {e}")
    
    def _get_from_cache(self, news_text: str) -> Optional[Dict]:
        """从缓存获取结果"""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(news_text)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        return None
    
    def _save_to_cache(self, news_text: str, result: Dict) -> None:
        """保存结果到缓存"""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(news_text)
        self.cache[cache_key] = result
        self._save_cache()
    
    def _keyword_analysis(self, news_text: str) -> Tuple[int, int, str]:
        """
        关键词分析（用于混合架构）
        
        Returns:
            (positive_count, negative_count, prediction)
        """
        text_lower = news_text.lower()
        
        positive_words = [
            'rise', 'gain', 'positive', 'strong', 'beat', 'up', 'growth',
            'increase', 'success', 'bull', 'soar', 'surge', 'rally', 'jump'
        ]
        negative_words = [
            'fall', 'drop', 'negative', 'weak', 'miss', 'down', 'decline',
            'decrease', 'failure', 'bear', 'crash', 'plunge', 'slump', 'tumble'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            prediction = "UP"
        elif negative_count > positive_count:
            prediction = "DOWN"
        else:
            prediction = "NEUTRAL"
        
        return positive_count, negative_count, prediction
    
    def _keyword_fallback(self, news_text: str) -> Dict:
        """Keyword fallback方法"""
        positive_count, negative_count, prediction = self._keyword_analysis(news_text)
        
        if prediction == "UP":
            confidence = min(1.0, 0.5 + 0.1 * (positive_count - negative_count))
            reason = f"Positive keywords detected: {positive_count} vs {negative_count}"
        elif prediction == "DOWN":
            confidence = min(1.0, 0.5 + 0.1 * (negative_count - positive_count))
            reason = f"Negative keywords detected: {negative_count} vs {positive_count}"
        else:
            random.seed(hash(news_text))
            prediction = "UP" if random.random() < 0.5 else "DOWN"
            confidence = 0.5
            reason = "Neutral news, random prediction"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "method": "keyword_fallback"
        }
    
    def _build_prompt(self, news_text: str) -> str:
        """构建prompt"""
        prompt = f"""You are a financial analyst.

Given the following news, predict whether the asset price will go UP or DOWN in the next trading day.

Constraints:
- Only use the information in the news
- Do NOT assume future information
- Output strictly in JSON format:

{{
  "prediction": "UP" or "DOWN",
  "confidence": 0~1,
  "reason": "brief explanation"
}}

News:
{news_text}
"""
        return prompt
    
    def _parse_output(self, llm_output: str) -> Dict:
        """解析LLM输出"""
        try:
            parsed = json.loads(llm_output)
            return parsed
        except json.JSONDecodeError:
            try:
                start_idx = llm_output.find('{')
                end_idx = llm_output.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = llm_output[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    return parsed
            except Exception:
                pass
            raise ValueError("无法解析LLM输出为JSON")
    
    def _call_llm_with_backoff(self, news_text: str, max_retries: int = 5) -> Optional[str]:
        """
        指数退避重试调用LLM
        
        Args:
            news_text: 新闻文本
            max_retries: 最大重试次数
        
        Returns:
            llm_output: LLM输出，失败返回None
        """
        if not self.client:
            return None
        
        prompt = self._build_prompt(news_text)
        
        for retry in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                if (response and hasattr(response, 'choices') and 
                    len(response.choices) > 0 and 
                    hasattr(response.choices[0], 'message') and 
                    hasattr(response.choices[0].message, 'content') and
                    response.choices[0].message.content is not None):
                    return response.choices[0].message.content.strip()
                else:
                    raise ValueError("LLM响应格式错误或为空")
                    
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait_time = 2 ** retry
                    print(f"  速率限制，等待 {wait_time} 秒 (retry {retry+1}/{max_retries})")
                    time.sleep(wait_time)
                elif retry < max_retries - 1:
                    wait_time = 2 ** retry
                    print(f"  LLM调用失败，等待 {wait_time} 秒 (retry {retry+1}/{max_retries}): {error_str[:100]}")
                    time.sleep(wait_time)
                else:
                    print(f"  LLM调用失败，已达最大重试次数: {error_str[:100]}")
        
        return None
    
    def llm_predict(self, news_text: str, max_retries: int = 5) -> Dict:
        """
        使用LLM预测新闻（优化版）
        
        Args:
            news_text: 新闻文本
            max_retries: 最大重试次数
        
        Returns:
            prediction_result: 预测结果
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        # 1. 尝试从缓存获取
        cached_result = self._get_from_cache(news_text)
        if cached_result is not None:
            result = cached_result.copy()
            result["method"] = "cache"
            return result
        
        # 2. 混合架构：关键词预过滤
        if self.use_hybrid_filter:
            positive_count, negative_count, keyword_prediction = self._keyword_analysis(news_text)
            diff = abs(positive_count - negative_count)
            
            if diff > self.hybrid_threshold:
                self.stats["hybrid_filter_hits"] += 1
                keyword_result = self._keyword_fallback(news_text)
                result = {
                    "signal": 1 if keyword_result["prediction"] == "UP" else -1,
                    "confidence": keyword_result["confidence"],
                    "reason": keyword_result["reason"],
                    "method": "hybrid_filter"
                }
                self._save_to_cache(news_text, result)
                return result
        
        # 3. 调用LLM
        method = "llm"
        llm_success = False
        llm_output = None
        error_message = None
        
        if self.client:
            llm_output = self._call_llm_with_backoff(news_text, max_retries=max_retries)
            if llm_output:
                llm_success = True
                self.stats["api_calls"] += 1
        
        # 4. 解析输出或fallback
        if llm_success and llm_output:
            try:
                parsed_output = self._parse_output(llm_output)
                
                if parsed_output["prediction"].upper() == "UP":
                    signal = 1
                elif parsed_output["prediction"].upper() == "DOWN":
                    signal = -1
                else:
                    raise ValueError(f"无效的prediction: {parsed_output['prediction']}")
                
                result = {
                    "signal": signal,
                    "confidence": float(parsed_output.get("confidence", 0.5)),
                    "reason": parsed_output.get("reason", ""),
                    "method": "llm"
                }
                
            except Exception as e:
                print(f"  LLM输出解析失败: {e}")
                if self.fallback_to_keyword:
                    print("  使用keyword fallback方法")
                    method = "keyword_fallback"
                    self.stats["fallback_calls"] += 1
                    keyword_result = self._keyword_fallback(news_text)
                    result = {
                        "signal": 1 if keyword_result["prediction"] == "UP" else -1,
                        "confidence": keyword_result["confidence"],
                        "reason": keyword_result["reason"],
                        "method": "keyword_fallback"
                    }
                else:
                    raise
        else:
            if self.fallback_to_keyword:
                print("  LLM不可用，使用keyword fallback方法")
                method = "keyword_fallback"
                self.stats["fallback_calls"] += 1
                keyword_result = self._keyword_fallback(news_text)
                result = {
                    "signal": 1 if keyword_result["prediction"] == "UP" else -1,
                    "confidence": keyword_result["confidence"],
                    "reason": keyword_result["reason"],
                    "method": "keyword_fallback"
                }
            else:
                raise ValueError("LLM不可用且fallback被禁用")
        
        self._save_to_cache(news_text, result)
        
        end_time = time.time()
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "news_text": news_text,
            "llm_output": llm_output,
            "error": error_message,
            "result": result,
            "method": method,
            "latency": end_time - start_time
        }
        self.logs.append(log_entry)
        
        return result
    
    async def _async_call_llm(self, news_text: str, semaphore: asyncio.Semaphore) -> Dict:
        """异步调用LLM（带信号量控制）"""
        async with semaphore:
            return self.llm_predict(news_text)
    
    async def batch_predict_async(self, news_texts: List[str]) -> List[Dict]:
        """异步批量预测"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._async_call_llm(text, semaphore) for text in news_texts]
        return await asyncio.gather(*tasks)
    
    def batch_predict(self, news_texts: List[str], use_async: bool = True) -> List[Dict]:
        """批量预测"""
        print(f"开始批量预测 ({len(news_texts)}条新闻)...")
        
        if use_async and self.async_client:
            print("  使用异步模式...")
            results = asyncio.run(self.batch_predict_async(news_texts))
        else:
            print("  使用同步模式...")
            results = []
            for i, text in enumerate(news_texts):
                print(f"  [{i+1}/{len(news_texts)}] 处理中...")
                results.append(self.llm_predict(text))
        
        print(f"批量预测完成！")
        self.print_stats()
        self.save_logs()
        
        return results
    
    def print_stats(self) -> None:
        """打印统计信息"""
        total = self.stats["total_calls"]
        cache_hits = self.stats["cache_hits"]
        hybrid_hits = self.stats["hybrid_filter_hits"]
        api_calls = self.stats["api_calls"]
        fallback_calls = self.stats["fallback_calls"]
        
        cache_hit_rate = (cache_hits / total * 100) if total > 0 else 0
        hybrid_hit_rate = (hybrid_hits / total * 100) if total > 0 else 0
        
        print()
        print("="*80)
        print("LLM调用统计（优化版）")
        print("="*80)
        print(f"总调用次数: {total}")
        print(f"缓存命中: {cache_hits} ({cache_hit_rate:.1f}%)")
        print(f"混合过滤命中: {hybrid_hits} ({hybrid_hit_rate:.1f}%)")
        print(f"API调用: {api_calls}")
        print(f"Fallback调用: {fallback_calls}")
        print("="*80)
        print()
    
    def save_logs(self) -> None:
        """保存日志"""
        try:
            existing_logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            
            all_logs = existing_logs + self.logs
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(all_logs, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 日志已保存到: {self.log_file}")
            
        except Exception as e:
            print(f"⚠️  日志保存失败: {e}")


if __name__ == '__main__':
    print("="*80)
    print("优化版LLM金融预测模块测试")
    print("="*80)
    print()
    
    test_news = [
        "Apple Inc. reports record quarterly earnings, beating Wall Street expectations on strong iPhone and Services growth.",
        "Apple supplier Foxconn reports production delays due to supply chain issues in China.",
        "Apple unveils new mixed-reality headset at annual developer conference.",
        "Apple Inc. reports record quarterly earnings, beating Wall Street expectations on strong iPhone and Services growth.",
        "Apple supplier Foxconn reports production delays due to supply chain issues in China."
    ]
    
    print(f"测试新闻数量: {len(test_news)}")
    print()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("KIMI_API_KEY")
    base_url = os.getenv("KIMI_BASE_URL")
    
    predictor = LLMFinancialPredictorOptimized(
        api_key=api_key,
        base_url=base_url,
        model="kimi-k2",
        fallback_to_keyword=True,
        use_cache=True,
        use_hybrid_filter=True,
        hybrid_threshold=0,
        max_concurrent=3
    )
    print()
    
    results = predictor.batch_predict(test_news, use_async=False)
    print()
    
    print("="*80)
    print("预测结果")
    print("="*80)
    print()
    
    for i, (news, result) in enumerate(zip(test_news, results)):
        print(f"新闻 {i+1}:")
        print(f"  文本: {news[:80]}...")
        print(f"  信号: {result['signal']}")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  原因: {result['reason']}")
        print(f"  方法: {result['method']}")
        print()
    
    print("="*80)
    print("测试完成！")
    print("="*80)
