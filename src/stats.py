from scipy import stats
from typing import List, Tuple


def perform_statistical_test(returns1: List[float], returns2: List[float]) -> Tuple[float, float, str]:
    """
    执行统计检验（自动选择参数或非参数检验）
    
    Args:
        returns1: 策略1的收益率列表
        returns2: 策略2的收益率列表
        
    Returns:
        statistic: 检验统计量
        p_value: p值
        test_type: 检验类型
    """
    if len(returns1) >= 20 and len(returns2) >= 20:
        try:
            _, p1 = stats.shapiro(returns1)
            _, p2 = stats.shapiro(returns2)
            
            if p1 < 0.05 or p2 < 0.05:
                test_stat, p_value = stats.wilcoxon(returns1, returns2)
                return test_stat, p_value, 'Wilcoxon'
            else:
                test_stat, p_value = stats.ttest_rel(returns1, returns2)
                return test_stat, p_value, 't-test'
        except Exception:
            test_stat, p_value = stats.wilcoxon(returns1, returns2)
            return test_stat, p_value, 'Wilcoxon'
    else:
        test_stat, p_value = stats.wilcoxon(returns1, returns2)
        return test_stat, p_value, 'Wilcoxon'
