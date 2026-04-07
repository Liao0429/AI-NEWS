import pandas as pd
import numpy as np
from src.backtest import run_backtest, print_backtest_results


# 创建测试数据
def create_test_data():
    """创建测试数据"""
    data = {
        'news_text': [
            'Apple announces record earnings',
            'iPhone sales decline',
            'New product launch',
            'Supply chain issues',
            'Positive analyst rating'
        ],
        'trade_price': [100.0, 101.0, 102.0, 103.0, 104.0],
        'future_price': [101.0, 100.5, 103.0, 102.5, 105.0]
    }
    return pd.DataFrame(data)


def main():
    """测试回测功能"""
    print("=" * 80)
    print("测试升级后的回测功能")
    print("=" * 80)
    
    # 创建测试数据
    aligned_df = create_test_data()
    print("测试数据:")
    print(aligned_df)
    print()
    
    # 运行回测
    results = run_backtest(aligned_df, num_runs=10, random_seed=42)
    
    # 打印结果
    print_backtest_results(results)
    
    # 验证结果结构
    print("验证结果结构:")
    print(f"AI毛收益数量: {len(results['ai_gross_returns'])}")
    print(f"AI净收益数量: {len(results['ai_net_returns'])}")
    print(f"Rule毛收益数量: {len(results['rule_gross_returns'])}")
    print(f"Rule净收益数量: {len(results['rule_net_returns'])}")
    print()
    
    # 验证净收益是否小于毛收益（因为有成本）
    ai_gross_avg = np.mean(results['ai_gross_returns'])
    ai_net_avg = np.mean(results['ai_net_returns'])
    rule_gross_avg = np.mean(results['rule_gross_returns'])
    rule_net_avg = np.mean(results['rule_net_returns'])
    
    print("验证成本影响:")
    print(f"AI净收益是否小于毛收益: {ai_net_avg < ai_gross_avg}")
    print(f"Rule净收益是否小于毛收益: {rule_net_avg < rule_gross_avg}")
    print()
    
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()