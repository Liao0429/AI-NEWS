"""
Academic Visualization System - Comprehensive Test Script
Tests all 5 phases of the new visualization system:
Phase 1: Academic Style Configuration
Phase 2: Statistical Analysis Module
Phase 3: Professional Chart Types
Phase 4: Multi-Subplot Layouts
Phase 5: Error Bars, Reference Lines, Batch Export
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import (
    AcademicStyleConfig, StyleManager, StatisticalAnalyzer,
    StatisticalResult, AcademicVisualizationManager,
    get_visualization_manager
)


def test_phase1_style_system():
    """Test Phase 1: Academic Style Configuration System"""
    print("\n" + "=" * 60)
    print("TEST Phase 1: Academic Style Configuration")
    print("=" * 60)
    
    config = AcademicStyleConfig()
    style = StyleManager(config)
    
    assert config.font_size_title == 14
    assert config.figure_dpi == 300
    assert 'primary' in config.colors
    assert len(config.colors) >= 15
    
    color = style.get_color('primary')
    assert color == '#2166AC'
    
    colors_5 = style.get_strategy_colors(5)
    assert len(colors_5) == 5
    
    colors_10 = style.get_strategy_colors(10)
    assert len(colors_10) == 10
    
    print("[PASS] Style configuration loaded correctly")
    print(f"[INFO] Primary color: {color}")
    print(f"[INFO] Strategy colors (5): {len(colors_5)} colors")
    print(f"[INFO] DPI: {config.figure_dpi}")
    return True


def test_phase2_statistical_analysis():
    """Test Phase 2: Statistical Analysis Module"""
    print("\n" + "=" * 60)
    print("TEST Phase 2: Statistical Analysis Module")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer()
    
    group1 = [0.1, 0.15, 0.08, 0.12, 0.18, 0.09, 0.14, 0.11, 0.16, 0.13,
               0.1, 0.15, 0.08, 0.12, 0.18, 0.09, 0.14, 0.11, 0.16, 0.13,
               0.12, 0.14, 0.11, 0.13, 0.15, 0.1, 0.13, 0.14, 0.12, 0.11]
    group2 = [0.05, 0.08, 0.03, 0.06, 0.09, 0.04, 0.07, 0.05, 0.08, 0.06,
               0.05, 0.08, 0.03, 0.06, 0.09, 0.04, 0.07, 0.05, 0.08, 0.06,
               0.05, 0.07, 0.04, 0.06, 0.08, 0.05, 0.06, 0.07, 0.05, 0.04]
    
    result = analyzer.paired_t_test_with_effect(group1, group2)
    
    assert isinstance(result, StatisticalResult)
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'effect_size')
    assert hasattr(result, 'ci_lower')
    assert hasattr(result, 'ci_upper')
    assert result.n1 == len(group1)
    assert result.n2 == len(group2)
    
    stars = result.get_significance_stars()
    label = result.get_significance_label()
    assert isinstance(stars, str)
    assert isinstance(label, str)
    assert 'p=' in label
    
    d = analyzer.cohen_d(group1, group2)
    assert isinstance(d, float)
    
    ci_l, ci_u = analyzer.bootstrap_ci(group1)
    assert ci_l < ci_u
    
    power = analyzer.calculate_power_analysis(0.5, 50)
    assert 0 <= power <= 1
    
    req_n = analyzer.required_sample_size(0.5, 0.8)
    assert req_n > 0
    
    corrected = analyzer.bonferroni_correction([0.01, 0.02, 0.03, 0.04, 0.05])
    assert len(corrected) == 5
    assert corrected[0] == 0.05
    
    one_sample = analyzer.one_sample_t_test(group1, null_value=0.5)
    assert isinstance(one_sample, StatisticalResult)
    
    print("[PASS] All statistical analysis functions work correctly")
    print(f"[INFO] Paired t-test p-value: {result.p_value:.4f}")
    print(f"[INFO] Effect size (Cohen's d): {result.effect_size:.4f}")
    print(f"[INFO] Test type: {result.test_type}")
    print(f"[INFO] Significance: {stars} | Label: {label}")
    print(f"[INFO] Bootstrap CI: [{ci_l:.4f}, {ci_u:.4f}]")
    print(f"[INFO] Power (d=0.5, n=50): {power:.1%}")
    print(f"[INFO] Required N for 80% power (d=0.5): {req_n}")
    return True


def test_phase3_chart_types():
    """Test Phase 3: Professional Chart Types"""
    print("\n" + "=" * 60)
    print("TEST Phase 3: Professional Chart Types")
    print("=" * 60)
    
    viz = AcademicVisualizationManager(output_dir='results/figures/test_academic')
    
    models = ['VADER\n(Lexicon)', 'BERT\n(DL)', 'LLM\n(No Filter)', 'LLM\n(Agg Filter)']
    accuracies = [16.67, 34.29, 54.44, 72.22]
    sample_sizes = [90, 40, 90, 36]
    p_values = [1.0, 1.0, 0.3829, 0.0077]
    errors = [8.0, 12.0, 9.0, 10.0]
    
    path1 = viz.plot_model_comparison(
        models=models,
        accuracies=accuracies,
        sample_sizes=sample_sizes,
        p_values=p_values,
        errors=errors,
        model_subtitles=['Lexicon', 'Deep Learning', '', '(65/35)'],
        title='Figure 1: Model Performance Comparison',
        output_name='test_model_comparison'
    )
    assert os.path.exists(path1)
    print(f"[PASS] Figure 1 (Model Comparison): {path1}")
    
    conditions = ['No Filter', 'Aggressive Filter']
    values = [55.0, 72.22]
    p_vals_ablation = [0.3829, 0.0077]
    
    path2 = viz.plot_ablation_study(
        conditions=conditions,
        values=values,
        p_values=p_vals_ablation,
        condition_labels=['No Filter', 'Aggressive Filter\n(65/35)'],
        improvement_pct=17.22,
        sample_info='35/65',
        title='Figure 2: Ablation Study - Impact of Aggressive Filtering'
    )
    assert os.path.exists(path2)
    print(f"[PASS] Figure 2 (Ablation Study): {path2}")
    
    path3 = viz.plot_risk_analysis(
        win_count=26, loss_count=10,
        win_return=18.56, loss_return=-8.32,
        win_rate=0.7222, profit_loss_ratio=2.23,
        drawdown_events=[
            {'name': 'Credit Suisse\nCollapse', 'drawdown': -45.0, 'color': '#E64B35'},
            {'name': 'Google\nLayoffs', 'drawdown': -12.0, 'color': '#F4A582'},
            {'name': 'Others', 'drawdown': -6.0, 'color': '#92C5DE'},
        ],
        max_drawdown_excluded=8.0,
        title='Figure 3: Risk Analysis'
    )
    assert os.path.exists(path3)
    print(f"[PASS] Figure 3 (Risk Analysis): {path3}")
    
    np.random.seed(42)
    scores_no_filter = np.random.beta(1.5, 2, 90) * 100
    scores_filtered = np.concatenate([
        np.random.beta(2, 1.5, 25) * 30 + 5,
        np.random.beta(2, 1.5, 11) * 20 + 80
    ])
    
    path4 = viz.plot_sample_distribution_filtering(
        scores_no_filter=scores_no_filter.tolist(),
        scores_filtered=scores_filtered.tolist(),
        threshold_low=35, threshold_high=65,
        retention_rate=55.6, filtered_count=40, original_count=90,
        title='Figure 5: Sample Distribution - Effect of Aggressive Filtering'
    )
    assert os.path.exists(path4)
    print(f"[PASS] Figure 5 (Sample Distribution): {path4}")
    
    path5 = viz.plot_power_analysis(
        observed_n=36,
        observed_effect=0.44,
        target_power=0.80,
        title='Figure 6: Statistical Power and Sample Size Analysis'
    )
    assert os.path.exists(path5)
    print(f"[PASS] Figure 6 (Power Analysis): {path5}")
    
    strategy_results = {
        'LLM': {'mean': 0.72, 'sharpe': 1.24, 'win_rate': 72.22, 
                'max_drawdown': -12.5, 'calmar_ratio': 5.76},
        'Hash': {'mean': 0.55, 'sharpe': 0.89, 'win_rate': 55.0,
                 'max_drawdown': -18.3, 'calmar_ratio': 3.01},
        'Keyword': {'mean': 0.48, 'sharpe': 0.62, 'win_rate': 48.0,
                    'max_drawdown': -22.1, 'calmar_ratio': 2.17},
        'Random': {'mean': 0.50, 'sharpe': 0.01, 'win_rate': 50.0,
                   'max_drawdown': -25.0, 'calmar_ratio': 2.00},
    }
    strategy_returns = {
        'LLM': np.random.normal(0.01, 0.05, 36).tolist(),
        'Hash': np.random.normal(-0.002, 0.06, 36).tolist(),
        'Keyword': np.random.normal(-0.01, 0.07, 36).tolist(),
        'Random': np.random.choice([-1, 1], 36).tolist(),
    }
    
    path6 = viz.plot_strategy_comparison_academic(
        strategy_results=strategy_results,
        strategy_returns=strategy_returns,
        baseline_strategy='Random',
        title='Strategy Performance Comparison with Statistics'
    )
    assert os.path.exists(path6)
    print(f"[PASS] Figure (Strategy Comparison Academic): {path6}")
    
    path7 = viz.plot_cumulative_returns_regime(strategy_returns)
    assert os.path.exists(path7)
    print(f"[PASS] Figure (Cumulative Returns Regime): {path7}")
    
    market_perf = {
        'LLM': {
            'Bull': {'win_rate': 72.58, 'mean_return': 1.85, 'count': 69},
            'Bear': {'win_rate': 21.43, 'mean_return': -2.31, 'count': 14},
            'Sideways': {'win_rate': 58.33, 'mean_return': 0.42, 'count': 12}
        },
        'Hash': {
            'Bull': {'win_rate': 55.07, 'mean_return': 0.82, 'count': 69},
            'Bear': {'win_rate': 42.86, 'mean_return': -1.15, 'count': 14},
            'Sideways': {'win_rate': 50.0, 'mean_return': 0.21, 'count': 12}
        }
    }
    
    path8 = viz.plot_market_condition_heatmap_enhanced(market_perf)
    assert os.path.exists(path8)
    print(f"[PASS] Figure (Market Condition Heatmap Enhanced): {path8}")
    
    print(f"\n[PASS] All {8} chart types generated successfully!")
    return True


def test_phase4_multi_subplot():
    """Test Phase 4: Multi-Subplot Layouts"""
    print("\n" + "=" * 60)
    print("TEST Phase 4: Multi-Subplot Layouts")
    print("=" * 60)
    
    viz = AcademicVisualizationManager(output_dir='results/figures/test_academic')
    
    fig, axes = viz._get_figure_and_axes((16, 10), nrows=2, ncols=3)
    assert len(axes) == 2
    assert len(axes[0]) == 3
    
    import matplotlib.pyplot as plt
    for i in range(2):
        for j in range(3):
            ax = axes[i][j]
            ax.plot([1, 2, 3], [1, 4, 9], linewidth=2)
            ax.set_title(f'Subplot ({i},{j})', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(viz.output_dir, 'test_multi_subplot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    assert os.path.exists(path)
    print(f"[PASS] Multi-subplot layout (2x3): {path}")
    
    fig2, (ax_a, ax_b) = viz._get_figure_and_axes((14, 6), nrows=1, ncols=2)
    ax_a.bar([1, 2, 3], [3, 1, 4], color=viz.style.get_color('primary'))
    ax_b.bar([1, 2, 3], [2, 5, 3], color=viz.style.get_color('positive'))
    ax_a.set_title('(A) Panel A')
    ax_b.set_title('(B) Panel B')
    plt.tight_layout()
    path2 = os.path.join(viz.output_dir, 'test_two_panel.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    assert os.path.exists(path2)
    print(f"[PASS] Two-panel layout: {path2}")
    return True


def test_phase5_batch_export():
    """Test Phase 5: Batch Export and Integration Features"""
    print("\n" + "=" * 60)
    print("TEST Phase 5: Error Bars, Reference Lines, Batch Export")
    print("=" * 60)
    
    viz = AcademicVisualizationManager(output_dir='results/figures/test_academic')
    
    import matplotlib.pyplot as plt
    fig, ax = viz._get_figure_and_axes((10, 6))
    
    x = [1, 2, 3, 4]
    y = [2.5, 3.8, 1.2, 4.5]
    yerr = [0.3, 0.5, 0.2, 0.4]
    
    bars = ax.bar(x, y, yerr=yerr, capsize=5, color=viz.style.get_color('primary'),
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    viz._add_reference_line(ax, 3.0, label='Threshold', linestyle='--', alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    viz._style_axis(ax, title='Error Bars & Reference Line Test',
                   xlabel='Category', ylabel='Value')
    
    paths = viz.batch_export_all_formats(fig, 'test_batch_export',
                                          formats=['png', 'pdf', 'svg'])
    
    assert 'png' in paths
    assert 'pdf' in paths
    assert 'svg' in paths
    assert os.path.exists(paths['png'])
    assert os.path.exists(paths['pdf'])
    assert os.path.exists(paths['svg'])
    
    plt.close(fig)
    
    print(f"[PASS] Batch export: PNG={os.path.basename(paths['png'])}, "
          f"PDF={os.path.basename(paths['pdf'])}, SVG={os.path.basename(paths['svg'])}")
    
    manifest_path = viz.generate_summary_json()
    assert os.path.exists(manifest_path)
    print(f"[PASS] JSON manifest: {manifest_path}")
    
    assert len(viz.generated_figures) > 0
    print(f"[INFO] Total figures tracked: {len(viz.generated_figures)}")
    return True


def test_backward_compatibility():
    """Test backward compatibility with old VisualizationManager interface"""
    print("\n" + "=" * 60)
    print("TEST Backward Compatibility Layer")
    print("=" * 60)
    
    from src.visualization import VisualizationManager
    
    old_viz = VisualizationManager(output_dir='results/figures/test_compat')
    
    sr = {
        'LLM': {'mean': 0.72, 'sharpe': 1.24, 'max_drawdown': -12.5,
                'win_rate': 72.22, 'calmar_ratio': 5.76, 'sortino_ratio': 1.8},
        'Hash': {'mean': 0.55, 'sharpe': 0.89, 'max_drawdown': -18.3,
                 'win_rate': 55.0, 'calmar_ratio': 3.01, 'sortino_ratio': 1.2}
    }
    sret = {
        'LLM': np.random.normal(0.01, 0.05, 30).tolist(),
        'Hash': np.random.normal(-0.002, 0.06, 30).tolist()
    }
    ea = {'Easy': {'count': 45}, 'Hard': {'count': 25}, 'LLM-special': {'count': 8}, 'Hash-special': {'count': 2}}
    mp = {
        'LLM': {'Bull': 72.58, 'Bear': 21.43, 'Sideways': 58.33},
        'Hash': {'Bull': 55.07, 'Bear': 42.86, 'Sideways': 50.0}
    }
    sd = {
        'Momentum': {'window_size': [5, 10, 15, 20], 'win_rate': [52, 58, 55, 51]},
        'MeanReversion': {'window_size': [5, 10, 15, 20], 'win_rate': [48, 52, 49, 46]}
    }
    
    try:
        path1 = old_viz.plot_strategy_comparison(sr)
        assert os.path.exists(path1)
        print(f"[PASS] plot_strategy_comparison (compat)")
        
        path2 = old_viz.plot_return_distribution(sret)
        assert os.path.exists(path2)
        print(f"[PASS] plot_return_distribution (compat)")
        
        path3 = old_viz.plot_radar_chart(sr)
        assert os.path.exists(path3)
        print(f"[PASS] plot_radar_chart (compat)")
        
        path4 = old_viz.plot_cumulative_returns(sret)
        assert os.path.exists(path4)
        print(f"[PASS] plot_cumulative_returns (compat)")
        
        path5 = old_viz.plot_error_analysis_pie(ea)
        assert os.path.exists(path5)
        print(f"[PASS] plot_error_analysis_pie (compat)")
        
        path6 = old_viz.plot_market_condition_heatmap(mp)
        assert os.path.exists(path6)
        print(f"[PASS] plot_market_condition_heatmap (compat)")
        
        path7 = old_viz.plot_sensitivity_curve(sd)
        assert os.path.exists(path7)
        print(f"[PASS] plot_sensitivity_curve (compat)")
        
        all_paths = old_viz.generate_all_visualizations(sr, sret, ea, mp, sd)
        assert len(all_paths) > 0
        print(f"[PASS] generate_all_visualizations (compat) -> {len(all_paths)} figures")
        
    except Exception as e:
        print(f"[FAIL] Backward compatibility error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("[PASS] Full backward compatibility confirmed!")
    return True


def main():
    """Run all tests"""
    print("#" * 70)
    print("#  ACADEMIC VISUALIZATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("#" * 70)
    
    results = {}
    
    try:
        results['Phase 1: Style'] = test_phase1_style_system()
        print(f"[DEBUG] Phase 1 result: {results['Phase 1: Style']}")
    except Exception as e:
        print(f"[FAIL] Phase 1: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 1: Style'] = False
    
    try:
        results['Phase 2: Stats'] = test_phase2_statistical_analysis()
        print(f"[DEBUG] Phase 2 result: {results['Phase 2: Stats']}")
    except Exception as e:
        print(f"[FAIL] Phase 2: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 2: Stats'] = False
    
    try:
        results['Phase 3: Charts'] = test_phase3_chart_types()
        print(f"[DEBUG] Phase 3 result: {results['Phase 3: Charts']}")
    except Exception as e:
        print(f"[FAIL] Phase 3: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 3: Charts'] = False
    
    try:
        results['Phase 4: Layout'] = test_phase4_multi_subplot()
        print(f"[DEBUG] Phase 4 result: {results['Phase 4: Layout']}")
    except Exception as e:
        print(f"[FAIL] Phase 4: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 4: Layout'] = False
    
    try:
        results['Phase 5: Export'] = test_phase5_batch_export()
        print(f"[DEBUG] Phase 5 result: {results['Phase 5: Export']}")
    except Exception as e:
        print(f"[FAIL] Phase 5: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 5: Export'] = False
    
    try:
        results['Backward Compat'] = test_backward_compatibility()
        print(f"[DEBUG] Backward Compat result: {results['Backward Compat']}")
    except Exception as e:
        print(f"[FAIL] Backward Compat: {e}")
        import traceback
        traceback.print_exc()
        results['Backward Compat'] = False
    
    print("\n" + "#" * 70)
    print("#  TEST SUMMARY")
    print("#" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {name}")
    
    print("-" * 70)
    print(f"  Result: {passed}/{total} test groups passed")
    
    if passed == total:
        print("  Status: ALL TESTS PASSED!")
    else:
        print(f"  Status: {total - passed} test group(s) FAILED")
    
    print("#" * 70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
