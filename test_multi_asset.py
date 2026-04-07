import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset_adapter import load_multi_asset_data

def test_multi_asset():
    """
    测试多资产数据加载
    """
    print('='*100)
    print('测试多资产数据加载')
    print('='*100)
    
    # 测试加载多资产数据
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
    multi_asset_data = load_multi_asset_data(tickers, sample_size=50)
    
    # 打印结果
    print('\n' + '='*100)
    print('多资产数据加载结果')
    print('='*100)
    
    for ticker, data in multi_asset_data.items():
        print(f'\n{"="*60}')
        print(f'资产: {ticker}')
        print(f'数据条数: {len(data)}')
        if len(data) > 0:
            print(f'日期范围: {data["news_date"].min()} to {data["news_date"].max()}')
        print(f'加载成功: {len(data) > 0}')
    
    print('\n' + '='*100)
    print('测试完成！')
    print('='*100)

if __name__ == '__main__':
    test_multi_asset()
