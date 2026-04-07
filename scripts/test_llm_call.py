"""
测试真实LLM调用的脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from src.llm_model import get_reproducible_llm

# 创建日志文件
log_file = f"results/logs/llm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log(message):
    """同时输出到控制台和文件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

def main():
    log("=" * 80)
    log("开始LLM调用测试")
    log("=" * 80)

    # 初始化LLM
    log("步骤1: 初始化LLM...")
    llm = get_reproducible_llm(
        prompt_version="PROMPT_V1",
        model="kimi-k2"
    )
    log(f"LLM模型: {llm.model}")
    log(f"API Key存在: {bool(llm.api_key)}")
    log(f"API Base URL: {llm.base_url}")
    log(f"Client状态: {'已初始化' if llm.client else '未初始化'}")

    # 测试单个新闻
    log("\n步骤2: 测试单个新闻...")
    test_news = "Bitcoin price surge expected due to positive market sentiment and regulatory clarity."

    log(f"新闻文本: {test_news[:100]}...")

    result = llm.predict(test_news)

    log(f"预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    log(f"调用方法: {result.get('method', 'unknown')}")

    # 测试批量预测
    log("\n步骤3: 测试批量预测（3条新闻）...")
    test_news_list = [
        "Bitcoin rises 5% after positive regulatory news from SEC.",
        "Crypto market drops sharply due to new tax regulations.",
        "Trading volume increases as investors await Fed decision."
    ]

    results = llm.batch_predict(test_news_list)

    for i, r in enumerate(results):
        log(f"新闻{i+1}: {r.get('method', 'unknown')} -> signal={r.get('signal')}, confidence={r.get('confidence', 0):.2f}")

    # 打印统计
    log("\n步骤4: LLM调用统计...")
    log(f"总调用次数: {llm.stats['total_calls']}")
    log(f"缓存命中: {llm.stats['cache_hits']}")
    log(f"API调用: {llm.stats['api_calls']}")
    log(f"Fallback调用: {llm.stats['fallback_calls']}")

    # 保存缓存
    log(f"\n步骤5: 保存缓存到 {llm.cache_file}...")
    llm._save_cache()
    log(f"缓存条数: {len(llm.cache)}")

    log("\n" + "=" * 80)
    log("LLM调用测试完成")
    log(f"日志文件: {log_file}")
    log("=" * 80)

if __name__ == '__main__':
    main()
