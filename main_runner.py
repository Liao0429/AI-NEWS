import sys
sys.path.append('.')

import yaml
import os
import logging
from datetime import datetime

from src.dataset_adapter import DataProcessor
from src.strategies import BaseStrategy, KeywordStrategy, HashStrategy, RandomStrategy
from src.strategy_momentum import MomentumStrategy
from src.strategy_meanreversion import MeanReversionStrategy
from src.llm_model import get_reproducible_llm
from src.backtest import run_backtest, print_backtest_results
from src.visualization import get_visualization_manager


class ExperimentRegistry:
    """实验注册表，管理所有策略和资产"""
    
    def __init__(self, config_path='config/config.yaml'):
        """初始化实验注册表"""
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        self.processor = DataProcessor()
        
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logging.error(f"配置文件加载失败: {e}")
            # 返回默认配置
            return {
                'random_seed': 42,
                'num_runs': 100,
                'experiment_setting': {
                    'sample_size': 150,
                    'test_assets': ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD'],
                    'transaction_cost': 0.001
                },
                'strategies': {
                    'llm': {
                        'model': 'kimi-k2',
                        'pure_mode': True,
                        'prompt_version': 'PROMPT_V1',
                        'use_cache': True
                    },
                    'hash': {
                        'seed': 42
                    },
                    'momentum': {
                        'time_window': 5
                    },
                    'mean_reversion': {
                        'time_window': 5
                    }
                },
                'asset': 'AAPL',
                'start_date': '2023-01-01',
                'end_date': '2024-12-31',
                'noise_rate': 0.02,
                'trading_cost': 0.001,
                'slippage': 0.0005,
                'max_position': 1,
                'news_data_path': 'data/real/real_news_dataset.csv',
                'results_path': 'results/tables/',
                'logs_path': 'results/logs/'
            }
    
    def _setup_logging(self):
        """设置日志"""
        logs_path = self.config.get('logs_path', 'results/logs/')
        os.makedirs(logs_path, exist_ok=True)
        
        log_file = os.path.join(logs_path, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_strategies(self):
        """获取所有策略实例"""
        # 获取策略配置
        strategy_configs = self.config.get('strategies', {})
        
        strategies = {
            'Keyword': KeywordStrategy(random_seed=self.config['random_seed']),
            'Hash': HashStrategy(random_seed=strategy_configs.get('hash', {}).get('seed', self.config['random_seed'])),
            'Momentum': MomentumStrategy(random_seed=self.config['random_seed']),
            'MeanReversion': MeanReversionStrategy(random_seed=self.config['random_seed']),
            'Random': RandomStrategy(random_seed=self.config['random_seed'])
        }
        
        # 添加LLM策略
        try:
            llm_config = strategy_configs.get('llm', {})
            llm = get_reproducible_llm(
                prompt_version=llm_config.get('prompt_version', 'PROMPT_V1'),
                model=llm_config.get('model', 'kimi-k2'),
                use_cache=llm_config.get('use_cache', True)
            )
            strategies['LLM'] = llm
            logging.info("LLM策略初始化成功")
        except Exception as e:
            logging.warning(f"LLM策略初始化失败: {e}")
        
        return strategies
    
    def load_data(self, asset=None):
        """加载数据"""
        asset = asset or self.config['asset']
        sample_size = self.config.get('experiment_setting', {}).get('sample_size', 150)
        logging.info(f"加载资产数据: {asset}, 样本大小: {sample_size}")
        
        try:
            df_trading = self.processor.load_from_news_to_forecast_data(
                ticker=asset,
                sample_size=sample_size  # 使用配置文件中的样本大小
            )
            logging.info(f"数据加载成功，共 {len(df_trading)} 条记录")
            return df_trading
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            raise
    
    def run_experiment(self, asset=None, strategies=None):
        """运行实验"""
        asset = asset or self.config['asset']
        strategies = strategies or list(self.get_strategies().keys())
        
        logging.info(f"开始实验: 资产={asset}, 策略={strategies}")
        
        # 加载数据
        df_trading = self.load_data(asset)
        
        # 运行回测
        results = run_backtest(
            df_trading,
            num_runs=self.config['num_runs'],
            random_seed=self.config['random_seed']
        )
        
        # 打印结果
        print_backtest_results(results)
        
        # 生成可视化
        # 注意：generate_all_visualizations 需要特定参数，暂时跳过
        # viz_manager = get_visualization_manager()
        # viz_manager.generate_all_visualizations()
        logging.info("可视化生成功能暂时跳过")
        
        logging.info("实验完成")
        return results
    
    def run_all_assets(self, assets=None):
        """运行所有资产的实验"""
        # 使用配置文件中的测试资产
        assets = assets or self.config.get('experiment_setting', {}).get('test_assets', ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD'])
        
        all_results = {}
        for asset in assets:
            logging.info(f"\n{'='*100}")
            logging.info(f"运行资产: {asset}")
            logging.info(f"{'='*100}")
            
            try:
                results = self.run_experiment(asset=asset)
                all_results[asset] = results
            except Exception as e:
                logging.error(f"资产 {asset} 实验失败: {e}")
        
        return all_results


def main():
    """主函数"""
    registry = ExperimentRegistry()
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='运行LLM新闻交易实验')
    parser.add_argument('--asset', type=str, help='资产代码')
    parser.add_argument('--all-assets', action='store_true', help='运行所有资产')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    if args.config:
        registry = ExperimentRegistry(config_path=args.config)
    
    if args.all_assets:
        registry.run_all_assets()
    elif args.asset:
        registry.run_experiment(asset=args.asset)
    else:
        # 默认运行配置文件中的资产
        registry.run_experiment()


if __name__ == '__main__':
    main()
