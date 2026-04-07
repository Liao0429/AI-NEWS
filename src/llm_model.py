import sys
sys.path.append('.')

import os
import json
import hashlib
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI library not installed, will use fallback method")


class ReproducibleLLM:
    """
    完全可复现的LLM模块
    
    特性：
    - 固定参数：temperature=0, top_p=1, max_tokens=200
    - Prompt版本化（从config/prompts.yaml加载）
    - 完整的输入输出缓存（data/cache/llm_cache.json）
    - 相同输入永远返回相同输出
    """
    
    FIXED_TEMPERATURE = 0.0
    FIXED_TOP_P = 1.0
    FIXED_MAX_TOKENS = 200
    DEFAULT_CACHE_FILE = "data/cache/llm_cache.json"
    DEFAULT_PROMPT_FILE = "config/prompts.yaml"
    DEFAULT_PROMPT_VERSION = "PROMPT_V1"
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "kimi-k2",
                 prompt_version: str = DEFAULT_PROMPT_VERSION,
                 use_cache: bool = True,
                 cache_file: str = DEFAULT_CACHE_FILE,
                 prompt_file: str = DEFAULT_PROMPT_FILE):
        """
        初始化可复现LLM模块
        
        Args:
            api_key: LLM API key
            base_url: API基础URL
            model: 模型名称
            prompt_version: Prompt版本（PROMPT_V1, PROMPT_V2）
            use_cache: 是否使用缓存
            cache_file: 缓存文件路径
            prompt_file: Prompt配置文件路径
        """
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("KIMI_BASE_URL", "https://apis.iflow.cn/v1")
        self.model = model
        self.prompt_version = prompt_version
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.prompt_file = prompt_file
        
        self.cache = {}
        self.prompts = {}
        self.client = None
        
        self._ensure_directories()
        self._load_prompts()
        self._load_cache()
        self._init_client()
        
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "parse_failures": 0
        }
    
    def _ensure_directories(self):
        """确保所有必要目录存在"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.prompt_file), exist_ok=True)
    
    def _load_prompts(self):
        """从YAML文件加载Prompt版本"""
        if os.path.exists(self.prompt_file):
            try:
                with open(self.prompt_file, 'r', encoding='utf-8') as f:
                    self.prompts = yaml.safe_load(f)
                print(f"[OK] Prompt loaded: {list(self.prompts.keys())}")
            except Exception as e:
                print(f"[WARNING] Prompt load failed: {e}")
                self._load_default_prompts()
        else:
            print(f"[WARNING] Prompt file not found: {self.prompt_file}")
            self._load_default_prompts()
    
    def _load_default_prompts(self):
        """加载默认Prompt作为fallback"""
        self.prompts = {
            "PROMPT_V1": {
                "name": "Financial Analyst V1",
                "description": "Basic financial news analysis prompt",
                "content": """You are a financial analyst.

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
{news_text}"""
            }
        }
    
    def _get_prompt(self, news_text: str) -> str:
        """获取指定版本的Prompt并插入新闻文本"""
        if self.prompt_version not in self.prompts:
            print(f"[WARNING] Prompt version not found: {self.prompt_version}, using PROMPT_V1")
            prompt_content = self.prompts["PROMPT_V1"]["content"]
        else:
            prompt_content = self.prompts[self.prompt_version]["content"]
        
        return prompt_content.format(news_text=news_text)
    
    def _get_cache_key(self, news_text: str) -> str:
        """生成稳定的缓存key（包含模型、prompt版本、新闻文本）"""
        key_components = f"{self.model}:{self.prompt_version}:{news_text}"
        return hashlib.md5(key_components.encode('utf-8')).hexdigest()
    
    def _load_cache(self):
        """加载缓存"""
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"[OK] Cache loaded ({len(self.cache)} entries)")
            except Exception as e:
                print(f"[WARNING] Cache load failed: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """保存缓存"""
        if self.use_cache:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARNING] Cache save failed: {e}")
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                print(f"[OK] LLM client initialized (model: {self.model}, prompt: {self.prompt_version})")
            except Exception as e:
                print(f"[WARNING] LLM client init failed: {e}")
                self.client = None
        else:
            print("[WARNING] LLM client not initialized, will use fallback method")
    
    def _parse_output(self, llm_output: str) -> Dict:
        """解析LLM输出为JSON"""
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            try:
                start_idx = llm_output.find('{')
                end_idx = llm_output.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    return json.loads(llm_output[start_idx:end_idx+1])
            except Exception:
                pass
            self.stats["parse_failures"] += 1
            raise ValueError("无法解析LLM输出为JSON")
    
    def _keyword_fallback(self, news_text: str) -> Dict:
        """Keyword fallback方法（固定种子确保可复现）"""
        text_lower = news_text.lower()
        
        positive_words = ['rise', 'gain', 'positive', 'strong', 'beat', 'up', 'growth',
                         'increase', 'success', 'bull', 'soar', 'surge', 'rally', 'jump']
        negative_words = ['fall', 'drop', 'negative', 'weak', 'miss', 'down', 'decline',
                         'decrease', 'failure', 'bear', 'crash', 'plunge', 'slump', 'tumble']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        import random
        random.seed(hash(news_text))
        
        if positive_count > negative_count:
            prediction = "UP"
            confidence = min(1.0, 0.5 + 0.1 * (positive_count - negative_count))
        elif negative_count > positive_count:
            prediction = "DOWN"
            confidence = min(1.0, 0.5 + 0.1 * (negative_count - positive_count))
        else:
            prediction = "UP" if random.random() < 0.5 else "DOWN"
            confidence = 0.5
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": f"Keyword fallback: pos={positive_count}, neg={negative_count}",
            "method": "keyword_fallback"
        }
    
    def predict(self, news_text: str) -> Dict:
        """
        预测新闻（完全可复现）
        
        Args:
            news_text: 新闻文本
            
        Returns:
            {
                "signal": +1, -1, or 0 (neutral),
                "confidence": float (0~1),
                "reason": string,
                "method": "cache" or "llm" or "neutral",
                "model_name": string,
                "prompt_version": string
            }
        """
        self.stats["total_calls"] += 1
        
        cache_key = self._get_cache_key(news_text)
        
        if self.use_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            result = self.cache[cache_key].copy()
            result["method"] = "cache"
            return result
        
        prompt = self._get_prompt(news_text)
        result = None
        method = None
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.FIXED_TEMPERATURE,
                    top_p=self.FIXED_TOP_P,
                    max_tokens=self.FIXED_MAX_TOKENS
                )
                
                if (response and hasattr(response, 'choices') and 
                    len(response.choices) > 0 and 
                    hasattr(response.choices[0], 'message') and 
                    hasattr(response.choices[0].message, 'content') and
                    response.choices[0].message.content is not None):
                    
                    llm_output = response.choices[0].message.content.strip()
                    parsed = self._parse_output(llm_output)
                    
                    signal = 1 if parsed["prediction"].upper() == "UP" else -1
                    
                    result = {
                        "signal": signal,
                        "confidence": float(parsed.get("confidence", 0.5)),
                        "reason": parsed.get("reason", ""),
                        "method": "llm",
                        "model_name": self.model,
                        "prompt_version": self.prompt_version
                    }
                    
                    self.stats["api_calls"] += 1
                    method = "llm"
                    
            except Exception as e:
                print(f"  [WARNING] LLM call failed: {e}")
        
        if result is None:
            # LLM 无法解析，记为 Neutral (0)
            result = {
                "signal": 0,
                "confidence": 0.0,
                "reason": "LLM 无法解析，记为中性",
                "method": "neutral",
                "model_name": self.model,
                "prompt_version": self.prompt_version
            }
            method = "neutral"
        
        if self.use_cache:
            self.cache[cache_key] = result
            self._save_cache()
        
        return result
    
    def batch_predict(self, news_texts: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for i, news in enumerate(news_texts):
            print(f"  [{i+1}/{len(news_texts)}] 处理中...")
            results.append(self.predict(news))
        return results
    
    def print_stats(self):
        """打印统计信息"""
        total = self.stats["total_calls"]
        cache_hits = self.stats["cache_hits"]
        cache_hit_rate = (cache_hits / total * 100) if total > 0 else 0
        parse_failures = self.stats.get("parse_failures", 0)
        parse_failure_rate = (parse_failures / total * 100) if total > 0 else 0
        
        print()
        print("="*80)
        print("LLM调用统计（可复现）")
        print("="*80)
        print(f"总调用次数: {total}")
        print(f"缓存命中: {cache_hits} ({cache_hit_rate:.1f}%)")
        print(f"API调用: {self.stats['api_calls']}")
        print(f"解析失败: {parse_failures} ({parse_failure_rate:.1f}%)")
        print(f"模型: {self.model}")
        print(f"Prompt版本: {self.prompt_version}")
        print("="*80)
        print()


def get_reproducible_llm(prompt_version: str = "PROMPT_V1", 
                          model: str = "kimi-k2",
                          use_cache: bool = True) -> ReproducibleLLM:
    """
    获取可复现LLM实例的便捷函数
    
    Args:
        prompt_version: Prompt版本
        model: 模型名称
        use_cache: 是否使用缓存
    
    Returns:
        ReproducibleLLM实例
    """
    return ReproducibleLLM(
        prompt_version=prompt_version,
        model=model,
        use_cache=use_cache
    )


if __name__ == '__main__':
    print("="*80)
    print("可复现LLM模块测试")
    print("="*80)
    print()
    
    test_news = [
        "Apple Inc. reports record quarterly earnings, beating Wall Street expectations.",
        "Apple supplier Foxconn reports production delays due to supply chain issues.",
        "Apple Inc. reports record quarterly earnings, beating Wall Street expectations.",
        "Apple supplier Foxconn reports production delays due to supply chain issues."
    ]
    
    print(f"测试新闻数量: {len(test_news)}")
    print(f"（注意：第3-4条与第1-2条相同，用于测试缓存）")
    print()
    
    llm = get_reproducible_llm(prompt_version="PROMPT_V1")
    print()
    
    print("开始预测...")
    results = llm.batch_predict(test_news)
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
        print(f"  模型: {result['model_name']}")
        print(f"  Prompt版本: {result['prompt_version']}")
        print()
    
    llm.print_stats()
    print("="*80)
    print("测试完成！")
    print("="*80)
