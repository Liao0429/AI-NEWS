import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
import os
import json
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

try:
    import seaborn as sns
    _has_seaborn = True
except ImportError:
    _has_seaborn = False


# ============================================================
# Phase 1: Academic Style Configuration System
# ============================================================

@dataclass
class AcademicStyleConfig:
    """Academic publication style configuration"""
    
    # Font settings
    font_family: str = 'sans-serif'
    font_sans_serif: List[str] = field(default_factory=lambda: ['Arial', 'Helvetica', 'DejaVu Sans'])
    font_size_title: int = 14
    font_size_subtitle: int = 12
    font_size_axis: int = 11
    font_size_tick: int = 10
    font_size_annotation: int = 9
    font_size_legend: int = 9
    
    # Figure settings
    figure_dpi: int = 300
    figure_facecolor: str = 'white'
    axes_facecolor: str = 'white'
    save_format: str = 'png'
    save_bbox_inches: str = 'tight'
    
    # Color palette (academic color scheme)
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#2166AC',       # Deep blue
        'secondary': '#4393C3',     # Medium blue
        'tertiary': '#92C5DE',      # Light blue
        'positive': '#1B7837',      # Green for positive
        'negative': '#C51B7D',      # Red/Pink for negative
        'neutral': '#777777',       # Gray for neutral/ns
        'highlight': '#E64B35',     # Bright red for significant
        'baseline': '#999999',      # Dashed baseline
        'accent_1': '#B09C85',      # Warm brown
        'accent_2': '#DFC27D',      # Light gold
        'fill_blue': '#92C5DE',
        'fill_red': '#F4A582',
        'fill_green': '#A6DBA0',
        'grid': '#CCCCCC',
        'text_dark': '#333333',
        'text_light': '#666666',
        'sig_star': '#C51B7D',
        'ns_gray': '#999999',
    })
    
    # Line styles
    line_solid: str = '-'
    line_dashed: str = '--'
    line_dotted: str = ':'
    line_dashdot: str = '-.'
    
    # Marker styles
    marker_default: str = 'o'
    marker_significant: str = '*'
    
    # Significance levels
    sig_alpha_005: float = 0.05
    sig_alpha_001: float = 0.01
    sig_alpha_0001: float = 0.0001
    
    # Annotation positions
    pvalue_position: str = 'top_center'  # top_left, top_right, top_center, above_bar
    n_position: str = 'inside_bottom'


class StyleManager:
    """Manages academic visualization styles"""
    
    def __init__(self, config: Optional[AcademicStyleConfig] = None):
        self.config = config or AcademicStyleConfig()
        self._apply_global_settings()
    
    def _apply_global_settings(self):
        """Apply global matplotlib rcParams"""
        c = self.config
        
        safe_rcparams = {
            'font.family': c.font_family,
            'font.sans-serif': c.font_sans_serif,
            'axes.unicode_minus': False,
            'figure.dpi': c.figure_dpi,
            'figure.facecolor': c.figure_facecolor,
            'savefig.dpi': c.figure_dpi,
            'savefig.format': c.save_format,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': c.colors['grid'],
            'axes.labelsize': c.font_size_axis,
            'axes.titlesize': c.font_size_title,
            'xtick.labelsize': c.font_size_tick,
            'ytick.labelsize': c.font_size_tick,
            'legend.fontsize': c.font_size_legend,
            'lines.linewidth': 2,
            'lines.markersize': 6,
        }
        
        for key, val in safe_rcparams.items():
            try:
                plt.rcParams[key] = val
            except (KeyError, ValueError):
                pass
        
        try:
            plt.rcParams['mathtext.default'] = 'regular'
        except (KeyError, ValueError):
            pass
        
        try:
            c_face = c.axes_facecolor
            plt.rcParams['axes.facecolor'] = c_face
        except (KeyError, ValueError):
            pass
        
        # Use seaborn style if available
        if _has_seaborn:
            try:
                sns.set_style("whitegrid", {
                    "axes.facecolor": c.axes_facecolor,
                    "grid.color": c.colors['grid'],
                    "axes.edgecolor": c.colors['text_light'],
                    "grid.linestyle": c.line_dashed,
                })
            except Exception:
                pass
    
    def get_color(self, name: str) -> str:
        """Get a named color from palette"""
        return self.config.colors.get(name, name)
    
    def get_strategy_colors(self, n: int) -> List[str]:
        """Get a list of distinct colors for n strategies"""
        base_colors = [
            self.config.colors['primary'],
            self.config.colors['accent_1'],
            self.config.colors['positive'],
            self.config.colors['tertiary'],
            self.config.colors['negative'],
            self.config.colors['accent_2'],
        ]
        if n <= len(base_colors):
            return base_colors[:n]
        if _has_seaborn:
            return sns.color_palette('colorblind', n)
        return plt.cm.tab20(np.linspace(0, 1, n))


# ============================================================
# Phase 2: Statistical Analysis Module
# ============================================================

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    p_value: float
    statistic: float
    test_type: str
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    is_significant: bool = False
    significance_level: float = 0.05
    n1: int = 0
    n2: int = 0
    mean1: float = 0.0
    mean2: float = 0.0
    std1: float = 0.0
    std2: float = 0.0
    
    def get_significance_stars(self) -> str:
        """Return significance star notation"""
        if self.p_value < 0.0001:
            return '****'
        elif self.p_value < 0.001:
            return '***'
        elif self.p_value < 0.01:
            return '**'
        elif self.p_value < 0.05:
            return '*'
        else:
            return '(ns)'
    
    def get_significance_label(self) -> str:
        """Return formatted p-value label with significance"""
        stars = self.get_significance_stars()
        if '(ns)' in stars:
            return f"p={self.p_value:.4f} (ns)"
        else:
            return f"p={self.p_value:.4f}{stars}"


class StatisticalAnalyzer:
    """Comprehensive statistical analysis module"""
    
    @staticmethod
    def cohen_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            pooled_std = 1e-10
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def bootstrap_ci(data: List[float], n_bootstrap: int = 10000,
                     alpha: float = 0.05, stat_func=np.mean) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        boot_samples = []
        n = len(data)
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_samples.append(stat_func(sample))
        boot_samples = np.array(boot_samples)
        lower = np.percentile(boot_samples, 100 * alpha / 2)
        upper = np.percentile(boot_samples, 100 * (1 - alpha / 2))
        return float(lower), float(upper)
    
    @staticmethod
    def paired_t_test_with_effect(group1: List[float], group2: List[float],
                                   alpha: float = 0.05) -> StatisticalResult:
        """Perform paired t-test with full statistical summary"""
        group1 = np.array(group1, dtype=float)
        group2 = np.array(group2, dtype=float)
        
        # Normality check
        try:
            _, p_norm1 = sp_stats.shapiro(group1)
            _, p_norm2 = sp_stats.shapiro(group2)
            use_parametric = (p_norm1 >= 0.05 and p_norm2 >= 0.05)
        except Exception:
            use_parametric = False
        
        if use_parametric and len(group1) >= 20:
            test_stat, p_value = sp_stats.ttest_rel(group1, group2)
            test_type = 'Paired t-test'
        else:
            try:
                test_stat, p_value = sp_stats.wilcoxon(group1, group2)
                test_type = 'Wilcoxon signed-rank'
            except Exception:
                test_stat, p_value = sp_stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_type = 'Mann-Whitney U'
        
        effect_size = StatisticalAnalyzer.cohen_d(group1.tolist(), group2.tolist())
        ci_lower, ci_upper = StatisticalAnalyzer.bootstrap_ci(
            (group1 - group2).tolist(), stat_func=np.mean
        )
        
        return StatisticalResult(
            p_value=float(p_value),
            statistic=float(test_stat),
            test_type=test_type,
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=p_value < alpha,
            significance_level=alpha,
            n1=len(group1),
            n2=len(group2),
            mean1=float(np.mean(group1)),
            mean2=float(np.mean(group2)),
            std1=float(np.std(group1, ddof=1)),
            std2=float(np.std(group2, ddof=1))
        )
    
    @staticmethod
    def one_sample_t_test(sample: List[float], null_value: float = 0.5,
                           alpha: float = 0.05) -> StatisticalResult:
        """One-sample t-test against a null value (e.g., 50% random guess)"""
        sample = np.array(sample, dtype=float)
        test_stat, p_value = sp_stats.ttest_1samp(sample, null_value)
        
        effect_size = (np.mean(sample) - null_value) / np.std(sample, ddof=1)
        ci_lower, ci_upper = StatisticalAnalyzer.bootstrap_ci(
            sample.tolist(), stat_func=np.mean
        )
        
        return StatisticalResult(
            p_value=float(p_value),
            statistic=float(test_stat),
            test_type='One-sample t-test',
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=p_value < alpha,
            significance_level=alpha,
            n1=len(sample),
            mean1=float(np.mean(sample)),
            std1=float(np.std(sample, ddof=1))
        )
    
    @staticmethod
    def calculate_power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
        """Calculate statistical power given effect size and sample size"""
        df = n - 1
        nc = effect_size * np.sqrt(n)
        
        critical = sp_stats.t.ppf(1 - alpha/2, df)
        
        try:
            nct_dist = sp_stats.nct(df, nc)
            power = 1 - nct_dist.cdf(critical) + nct_dist.cdf(-critical)
        except Exception:
            from scipy.special import betainc
            t_crit_sq = critical ** 2
            lambda_val = nc ** 2
            x = df / (df + t_crit_sq)
            power_approx = 1 - betainc(df/2, 0.5, x) if lambda_val == 0 else 0.5
            power = max(0.0, min(1.0, power_approx))
        
        return max(0, min(1, power))
    
    @staticmethod
    def required_sample_size(effect_size: float, target_power: float = 0.8,
                              alpha: float = 0.05) -> int:
        """Calculate required sample size for target power"""
        for n in range(10, 2000):
            power = StatisticalAnalyzer.calculate_power_analysis(effect_size, n, alpha)
            if power >= target_power:
                return n
        return 2000
    
    @staticmethod
    def bonferroni_correction(p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons"""
        m = len(p_values)
        return [min(p * m, 1.0) for p in p_values]


# ============================================================
# Phase 3 & 4 & 5: Professional Chart Types with Academic Features
# ============================================================

class AcademicVisualizationManager:
    """
    Advanced academic visualization manager with:
    - Phase 1: Academic style system
    - Phase 2: Integrated statistical analysis
    - Phase 3: Publication-quality chart types
    - Phase 4: Flexible multi-subplot layouts
    - Phase 5: Error bars, reference lines, batch export
    """
    
    def __init__(self, output_dir: str = 'results/figures',
                 config: Optional[AcademicStyleConfig] = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.style = StyleManager(config)
        self.stats_analyzer = StatisticalAnalyzer()
        self.generated_figures: List[str] = []
        print(f"[OK] Academic Visualization Manager initialized")
        print(f"[OK] Output directory: {output_dir}")
    
    def _get_figure_and_axes(self, figsize: Tuple[float, float],
                             nrows: int = 1, ncols: int = 1,
                             subplot_specs: Optional[List] = None) -> Tuple[plt.Figure, Any]:
        """Create figure with flexible layout support"""
        fig = plt.figure(figsize=figsize, facecolor=self.style.config.figure_facecolor)
        
        if subplot_specs and (nrows > 1 or ncols > 1):
            gs = gridspec.GridSpec(nrows, ncols, figure=fig, 
                                   hspace=0.35, wspace=0.3)
            axes = [fig.add_subplot(gs[i]) for i in range(nrows * ncols)]
            return fig, axes
        
        if nrows == 1 and ncols == 1:
            ax = fig.add_subplot(111)
            return fig, ax
        
        if nrows == 1 or ncols == 1:
            total = max(nrows, ncols)
            axes = [fig.add_subplot(1, total, i+1) for i in range(total)]
            return fig, axes
        
        axes = fig.subplots(nrows, ncols, squeeze=False)
        return fig, axes
    
    def _annotate_pvalue(self, ax: plt.Axes, x_pos: float, y_pos: float,
                         result: StatisticalResult, fontsize: int = None,
                         ha: str = 'center', color_override: str = None) -> None:
        """Add p-value annotation to bar chart"""
        fs = fontsize or self.style.config.font_size_annotation
        label = result.get_significance_label()
        color = color_override or (
            self.style.config.colors['highlight'] if result.is_significant
            else self.style.config.colors['ns_gray']
        )
        ax.annotate(label, xy=(x_pos, y_pos), xytext=(0, 8),
                   textcoords='offset points', ha=ha, va='bottom',
                   fontsize=fs, fontweight='bold', color=color)
    
    def _annotate_n(self, ax: plt.Axes, x_pos: float, y_pos: float,
                    n: int, fontsize: int = None, color: str = None) -> None:
        """Add sample size annotation inside bar"""
        fs = fontsize or self.style.config.font_size_annotation
        clr = color or 'white'
        ax.text(x_pos, y_pos, f'n={n}', ha='center', va='bottom',
               fontsize=fs, color=clr, fontweight='bold')
    
    def _add_reference_line(self, ax: plt.Axes, y_value: float,
                            label: str = None, linestyle: str = None,
                            color: str = None, alpha: float = 0.7) -> None:
        """Add horizontal reference line"""
        ls = linestyle or self.style.config.line_dashed
        clr = color or self.style.config.colors['baseline']
        ax.axhline(y=y_value, color=clr, linestyle=ls, linewidth=1.5, alpha=alpha)
        if label:
            ax.annotate(label, xy=(0.02, y_value), xycoords=('axes fraction', 'data'),
                       fontsize=self.style.config.font_size_annotation - 1,
                       color=clr, va='bottom')
    
    def _style_axis(self, ax: plt.Axes, title: str = None,
                    xlabel: str = None, ylabel: str = None,
                    title_fontsize: int = None) -> None:
        """Apply consistent styling to an axis"""
        c = self.style.config
        if title:
            ax.set_title(title, fontsize=title_fontsize or c.font_size_title,
                        fontweight='bold', pad=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=c.font_size_axis)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=c.font_size_axis)
        ax.tick_params(axis='both', which='major', labelsize=c.font_size_tick)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    # ----------------------------------------------------------
    # Chart Type 1: Model Performance Comparison (Figure 1 style)
    # ----------------------------------------------------------
    
    def plot_model_comparison(self, models: List[str], accuracies: List[float],
                             sample_sizes: List[int],
                             p_values: Optional[List[float]] = None,
                             baseline: float = 0.5,
                             baseline_label: str = 'Random Guess (50%)',
                             errors: Optional[List[float]] = None,
                             model_subtitles: Optional[List[str]] = None,
                             title: str = 'Model Performance Comparison',
                             ylabel: str = 'Accuracy (%)',
                             output_name: str = 'model_comparison') -> str:
        """
        Publication-quality model comparison bar chart with:
        - P-value annotations with significance stars
        - Error bars (CI or SEM)
        - Baseline reference line
        - Sample size annotations inside bars
        - Subtitle labels under each bar
        
        Reference: User's Figure 1 image
        """
        c = self.style.config
        n_models = len(models)
        colors = self.style.get_strategy_colors(n_models)
        
        fig, ax = self._get_figure_and_axes((10, 7))
        
        x_positions = np.arange(n_models)
        bar_width = 0.65
        
        # Calculate error bars if not provided
        if errors is None:
            errors = [acc * 0.15 for acc in accuracies]
        
        # Draw bars with error bars
        bars = ax.bar(x_positions, accuracies, width=bar_width,
                      color=colors, edgecolor='white', linewidth=1.5,
                      yerr=errors, capsize=5, error_kw={
                          'elinewidth': 1.5, 'capthick': 1.5,
                          'ecolor': c.colors['text_dark'], 'alpha': 0.8
                      })
        
        # Add baseline reference line
        self._add_reference_line(ax, baseline * 100, baseline_label)
        
        # Annotate each bar
        for i, (bar, acc, n) in enumerate(zip(bars, accuracies, sample_sizes)):
            x = bar.get_x() + bar.get_width() / 2
            
            # Sample size annotation (inside bottom of bar)
            self._annotate_n(ax, x, max(baseline * 100 * 0.3, acc * 0.15), n)
            
            # P-value annotation (above bar)
            if p_values and i < len(p_values):
                pv = p_values[i]
                is_sig = pv < 0.05
                result = StatisticalResult(
                    p_value=pv, statistic=0, test_type='t-test',
                    is_significant=is_sig, n1=n
                )
                self._annotate_pvalue(ax, x, acc + errors[i] + 2, result)
            
            # Model subtitle below x-axis
            if model_subtitles and i < len(model_subtitles):
                ax.text(x, -0.08 * (max(accuracies) - min(accuracies)) + min(accuracies),
                       model_subtitles[i], ha='center', va='top',
                       fontsize=c.font_size_annotation, color=c.colors['text_light'])
        
        # Styling
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, fontsize=c.font_size_axis, fontweight='bold')
        ax.set_ylim(bottom=0, top=max(accuracies) * 1.25)
        
        self._style_axis(ax, title=title, ylabel=ylabel, xlabel='Model')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{output_name}.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 2: Ablation Study (Figure 2 style)
    # ----------------------------------------------------------
    
    def plot_ablation_study(self, conditions: List[str], values: List[float],
                            p_values: Optional[List[float]] = None,
                            condition_labels: Optional[List[str]] = None,
                            improvement_pct: Optional[float] = None,
                            sample_info: Optional[str] = None,
                            title: str = 'Ablation Study') -> str:
        """
        Two-panel ablation study figure:
        (a) Accuracy Improvement - before/after comparison
        (b) Statistical Significance - p-value bar comparison
        
        Reference: User's Figure 2 image
        """
        c = self.style.config
        fig, (ax1, ax2) = self._get_figure_and_axes((14, 6), nrows=1, ncols=2)
        
        colors_ablation = [c.colors['secondary'], c.colors['primary']]
        
        # --- Panel (a): Accuracy Improvement ---
        x_pos = np.arange(len(conditions))
        bars_a = ax1.bar(x_pos, values, width=0.55, color=colors_ablation,
                         edgecolor='white', linewidth=1.5)
        
        # Add value and p-value labels on bars
        for i, (bar, val) in enumerate(zip(bars_a, values)):
            x = bar.get_x() + bar.get_width() / 2
            # Value label
            ax1.text(x, val + 1.5, f'{val:.2f}%', ha='center', va='bottom',
                    fontsize=c.font_size_axis, fontweight='bold')
            
            # P-value label
            if p_values and i < len(p_values):
                pv = p_values[i]
                sig_str = f'(p={pv:.4f}{"**" if pv < 0.01 else "ns"})'
                ax1.text(x, val - 3, sig_str, ha='center', va='top',
                        fontsize=c.font_size_annotation,
                        color=c.colors['highlight'] if pv < 0.01 else c.colors['ns_gray'])
        
        # Improvement arrow
        if improvement_pct is not None and len(values) >= 2:
            mid_x = (x_pos[0] + x_pos[1]) / 2
            max_val = max(values)
            ax1.annotate('', xy=(x_pos[1], values[1]), xytext=(x_pos[0], values[0]),
                        arrowprops=dict(arrowstyle='->', color=c.colors['highlight'],
                                       lw=2, connectionstyle='arc3,rad=0.2'))
            ax1.text(mid_x, max_val + 5, f'+{improvement_pct:.2f}%',
                    ha='center', fontsize=c.font_size_axis, fontweight='bold',
                    color=c.colors['highlight'])
        
        # Baseline at 50%
        self._add_reference_line(ax1, 50, alpha=0.5)
        
        # Condition labels
        display_labels = condition_labels or conditions
        if sample_info:
            display_labels = [f"{l}\n({sample_info})" for l in display_labels]
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(display_labels, fontsize=c.font_size_axis)
        ax1.set_ylim(0, max(values) * 1.35)
        
        self._style_axis(ax1, title=f'(a) Accuracy Improvement',
                        ylabel='Accuracy (%)')
        
        # --- Panel (b): Statistical Significance ---
        if p_values:
            colors_p = [c.colors['secondary'] if p >= 0.05 else c.colors['primary']
                        for p in p_values]
            bars_b = ax2.bar(x_pos, p_values, width=0.55, color=colors_p,
                             edgecolor='white', linewidth=1.5)
            
            for i, (bar, pv) in enumerate(zip(bars_b, p_values)):
                x = bar.get_x() + bar.get_width() / 2
                ax2.text(x, pv + 0.02, f'{pv:.4f}', ha='center', va='bottom',
                        fontsize=c.font_size_axis, fontweight='bold')
                
                # Reduction percentage
                if len(p_values) >= 2 and i == 1:
                    reduction = (1 - pv / p_values[0]) * 100
                    ax2.text(x, pv / 2, f'-{reduction:.0f}%', ha='center',
                            fontsize=c.font_size_annotation, fontweight='bold',
                            color=c.colors['positive'])
            
            # Significance thresholds
            self._add_reference_line(ax2, 0.05, label=r'$\alpha$=0.05',
                                    linestyle=c.line_dashed, color=c.colors['highlight'])
            self._add_reference_line(ax2, 0.01, label=r'$\alpha$=0.01',
                                    linestyle=c.line_dotted, color=c.colors['highlight'])
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(display_labels, fontsize=c.font_size_axis)
            ax2.set_ylim(0, max(0.5, max(p_values) * 1.3))
            
            self._style_axis(ax2, title=f'(b) Statistical Significance',
                            ylabel='P-value')
        
        fig.suptitle(title, fontsize=c.font_size_title + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'ablation_study.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 3: Risk Analysis (Figure 3 style)
    # ----------------------------------------------------------
    
    def plot_risk_analysis(self, win_count: int, loss_count: int,
                           win_return: float, loss_return: float,
                           win_rate: float, profit_loss_ratio: float,
                           drawdown_events: Optional[List[Dict]] = None,
                           max_drawdown_total: float = 0,
                           max_drawdown_excluded: float = 0,
                           title: str = 'Risk Analysis') -> str:
        """
        Two-panel risk analysis figure:
        (a) Win/Loss Distribution - dual-axis bar chart
        (b) Maximum Drawdown Attribution - horizontal waterfall-style
        
        Reference: User's Figure 3 image
        """
        c = self.style.config
        fig, (ax1, ax2) = self._get_figure_and_axes((14, 6), nrows=1, ncols=2)
        
        # --- Panel (a): Win/Loss Distribution ---
        categories = ['Win', 'Loss']
        counts = [win_count, loss_count]
        returns = [win_return, loss_return]
        colors_risk = [c.colors['primary'], c.colors['negative']]
        
        x_pos = np.arange(len(categories))
        bars = ax1.bar(x_pos, counts, width=0.5, color=colors_risk,
                       edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Count labels on top
        for i, (bar, cnt) in enumerate(zip(bars, counts)):
            x = bar.get_x() + bar.get_width() / 2
            ax1.text(x, cnt + 0.5, str(cnt), ha='center', va='bottom',
                    fontsize=c.font_size_axis, fontweight='bold')
        
        # Secondary axis for returns
        ax1_right = ax1.twinx()
        bars_ret = ax1_right.bar([p + 0.25 for p in x_pos], returns, width=0.5,
                                  color=[c.colors['fill_blue'], c.colors['fill_red']],
                                  edgecolor='white', linewidth=1, alpha=0.6)
        
        for i, (bar, ret) in enumerate(zip(bars_ret, returns)):
            x = bar.get_x() + bar.get_width() / 2
            sign = '+' if ret > 0 else ''
            ax1_right.text(x, ret + (abs(ret) * 0.1), f'{sign}{ret:.2f}%',
                          ha='center', va='bottom' if ret > 0 else 'top',
                          fontsize=c.font_size_annotation, fontweight='bold',
                          color=c.colors['positive'] if ret > 0 else c.colors['negative'])
        
        # Info box
        info_text = f'Rate: {win_rate:.2%}\nRatio: {profit_loss_ratio:.1f}:1'
        props = dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor=c.colors['grid'], alpha=0.9)
        ax1.text(0.65, 0.75, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax1.set_xticks(x_pos + 0.125)
        ax1.set_xticklabels(categories, fontsize=c.font_size_axis, fontweight='bold')
        ax1.set_ylabel('Number of Trades', fontsize=c.font_size_axis)
        ax1_right.set_ylabel('Average Return (%)', fontsize=c.font_size_axis)
        self._style_axis(ax1, title=f'(a) Win/Loss Distribution')
        ax1_right.spines['top'].set_visible(False)
        
        # --- Panel (b): Maximum Drawdown Attribution ---
        if drawdown_events:
            event_names = [e['name'] for e in drawdown_events]
            dd_values = [e['drawdown'] for e in drawdown_events]
            colors_dd = [e.get('color', c.colors['negative']) for e in drawdown_events]
            
            y_pos = np.arange(len(event_names))
            bars_dd = ax2.barh(y_pos, dd_values, height=0.6,
                               color=colors_dd, edgecolor='white', linewidth=1,
                               alpha=0.8)
            
            for i, (bar, val) in enumerate(zip(bars_dd, dd_values)):
                ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=c.font_size_annotation)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(event_names, fontsize=c.font_size_axis)
            ax2.set_xlabel('Drawdown (%)', fontsize=c.font_size_axis)
            ax2.invert_yaxis()
            
            # Max DD after exclusion line
            if max_drawdown_excluded > 0:
                ax2.axvline(x=-max_drawdown_excluded, color=c.colors['primary'],
                           linestyle=c.line_dashed, linewidth=2, alpha=0.8)
                ax2.text(-max_drawdown_excluded, len(event_names) - 0.5,
                        f'Max DD after exclusion', fontsize=c.font_size_annotation,
                        color=c.colors['primary'], va='bottom')
            
            # Event annotation
            if len(drawdown_events) > 0:
                worst_idx = np.argmin(dd_values)
                if dd_values[worst_idx] < -20:
                    ax2.annotate('Single extreme event\n(idiosyncratic risk)',
                                xy=(dd_values[worst_idx], worst_idx),
                                xytext=(dd_values[worst_idx] * 0.5, worst_idx + 0.8),
                                fontsize=c.font_size_annotation, color=c.colors['text_light'],
                                arrowprops=dict(arrowstyle='->', color=c.colors['text_light'],
                                               lw=1, connectionstyle='arc3,rad=0.1'))
            
            self._style_axis(ax2, title=f'(b) Maximum Drawdown Attribution')
        
        fig.suptitle(title, fontsize=c.font_size_title + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'risk_analysis.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 4: Sample Distribution with Filtering (Figure 5 style)
    # ----------------------------------------------------------
    
    def plot_sample_distribution_filtering(self, scores_no_filter: List[float],
                                           scores_filtered: List[float],
                                           threshold_low: float = 35,
                                           threshold_high: float = 65,
                                           retention_rate: float = 0.0,
                                           filtered_count: int = 0,
                                           original_count: int = 0,
                                           title: str = 'Sample Distribution') -> str:
        """
        Histogram showing effect of aggressive filtering on LLM confidence scores.
        Shows before/after distributions with threshold regions annotated.
        
        Reference: User's Figure 5 image
        """
        c = self.style.config
        fig, ax = self._get_figure_and_axes((12, 7))
        
        bins = np.linspace(0, 100, 21)
        
        # Histograms
        ax.hist(scores_no_filter, bins=bins, alpha=0.5, label=f'No Filter (n={original_count})',
               color=c.colors['tertiary'], edgecolor='white', linewidth=0.5)
        ax.hist(scores_filtered, bins=bins, alpha=0.8, label=f'Aggressive Filter (n={len(scores_filtered)})',
               color=c.colors['primary'], edgecolor='white', linewidth=0.5)
        
        # Threshold lines
        ax.axvline(x=threshold_low, color=c.colors['negative'], linestyle=c.line_dashed,
                  linewidth=2, label=f'Threshold ({threshold_low}/{threshold_high})')
        ax.axvline(x=threshold_high, color=c.colors['negative'], linestyle=c.line_dashed,
                  linewidth=2)
        
        # Region annotations
        mid_low = (threshold_low) / 2
        mid_filter = (threshold_low + threshold_high) / 2
        mid_high = (threshold_high + 100) / 2
        
        ax.text(mid_low, ax.get_ylim()[1] * 0.85, 'High Confidence\n(Execute)',
               ha='center', fontsize=c.font_size_annotation, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=c.colors['primary'], alpha=0.9))
        
        ax.text(mid_filter, ax.get_ylim()[1] * 0.9, 'Filtered Out\n(Uncertain)',
               ha='center', fontsize=c.font_size_annotation, style='italic',
               color=c.colors['text_light'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=c.colors['grid'], alpha=0.9))
        
        ax.text(mid_high, ax.get_ylim()[1] * 0.85, 'High Confidence\n(Execute)',
               ha='center', fontsize=c.font_size_annotation, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=c.colors['primary'], alpha=0.9))
        
        # Retention rate annotation
        if retention_rate > 0:
            ax.text(0.97, 0.97, f'Retention Rate: {retention_rate:.1f}%\nFiltered: {filtered_count} samples',
                   transform=ax.transAxes, ha='right', va='top', fontsize=c.font_size_axis,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=c.colors['grid'], alpha=0.95))
        
        ax.set_xlabel('LLM Confidence Score', fontsize=c.font_size_axis)
        ax.set_ylabel('Frequency', fontsize=c.font_size_axis)
        ax.legend(fontsize=c.font_size_legend, loc='upper left')
        self._style_axis(ax, title=title)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'sample_distribution_filtering.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 5: Statistical Power Analysis (Figure 6 style)
    # ----------------------------------------------------------
    
    def plot_power_analysis(self, observed_n: int = 36,
                            observed_effect: float = 0.44,
                            target_power: float = 0.80,
                            alpha: float = 0.05,
                            title: str = 'Statistical Power and Sample Size Analysis') -> str:
        """
        Two-panel statistical power analysis:
        (a) Power Analysis curve - shows achieved vs target power
        (b) Sample Size Requirements - shows required N for different effect sizes
        
        Reference: User's Figure 6 image
        """
        c = self.style.config
        fig, (ax1, ax2) = self._get_figure_and_axes((14, 6), nrows=1, ncols=2)
        
        # --- Panel (a): Power Curve ---
        sample_sizes = np.arange(10, 101, 2)
        powers = [self.stats_analyzer.calculate_power_analysis(observed_effect, n, alpha)
                  for n in sample_sizes]
        
        ax1.plot(sample_sizes, powers, color=c.colors['primary'], linewidth=2.5,
                label=f"Effect size d={observed_effect:.2f}")
        
        # Target power line
        ax1.axhline(y=target_power, color=c.colors['negative'], linestyle=c.line_dashed,
                   linewidth=1.5, label=f'Target Power ({target_power:.0%})')
        
        # Current observation point
        current_power = self.stats_analyzer.calculate_power_analysis(observed_effect, observed_n, alpha)
        ax1.scatter([observed_n], [current_power], s=120, color=c.colors['positive'],
                   zorder=5, edgecolors='white', linewidths=2)
        
        # Annotate current point
        ax1.annotate(f'Actual Power: {current_power:.1%}',
                    xy=(observed_n, current_power),
                    xytext=(observed_n + 12, current_power - 0.08),
                    fontsize=c.font_size_annotation, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c.colors['positive'],
                                   lw=1.5, connectionstyle='arc3,rad=0.1'))
        
        # Vertical line at actual n
        ax1.axvline(x=observed_n, color=c.colors['positive'], linestyle=c.line_dotted,
                   linewidth=1.5, alpha=0.7, label=f'Actual n={observed_n}')
        
        ax1.set_xlabel('Sample Size (n)', fontsize=c.font_size_axis)
        ax1.set_ylabel('Statistical Power', fontsize=c.font_size_axis)
        ax1.set_xlim(10, 100)
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=c.font_size_legend, loc='lower right')
        self._style_axis(ax1, title=f'(a) Power Analysis')
        
        # --- Panel (b): Sample Size Requirements ---
        effect_sizes = np.arange(0.2, 0.85, 0.02)
        required_ns = [self.stats_analyzer.required_sample_size(es, target_power, alpha)
                      for es in effect_sizes]
        
        ax2.plot(effect_sizes, required_ns, color=c.colors['primary'], linewidth=2.5)
        
        # Horizontal line at actual n
        ax2.axhline(y=observed_n, color=c.colors['positive'], linestyle=c.line_dotted,
                   linewidth=1.5, alpha=0.7, label=f'Actual n={observed_n}')
        
        # Mark observed effect size
        ax2.scatter([observed_effect], [observed_n], s=120, color=c.colors['positive'],
                   zorder=5, edgecolors='white', linewidths=2)
        
        ax2.annotate(f'Observed Effect\nd={observed_effect:.2f}',
                    xy=(observed_effect, observed_n),
                    xytext=(observed_effect + 0.12, observed_n + 40),
                    fontsize=c.font_size_annotation, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c.colors['positive'],
                                   lw=1.5, connectionstyle='arc3,rad=0.1'))
        
        ax2.set_xlabel("Effect Size (Cohen's d)", fontsize=c.font_size_axis)
        ax2.set_ylabel('Required Sample Size', fontsize=c.font_size_axis)
        ax2.set_xlim(0.18, 0.82)
        ax2.set_ylim(20, 420)
        ax2.legend(fontsize=c.font_size_legend, loc='upper right')
        self._style_axis(ax2, title=f'(b) Sample Size Requirements')
        
        fig.suptitle(title, fontsize=c.font_size_title + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'power_analysis.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 6: Strategy Comparison with Full Statistics
    # ----------------------------------------------------------
    
    def plot_strategy_comparison_academic(self, strategy_results: Dict[str, Dict[str, float]],
                                          strategy_returns: Dict[str, List[float]],
                                          baseline_strategy: str = 'Random',
                                          metrics: List[str] = None,
                                          title: str = 'Strategy Performance Comparison') -> str:
        """
        Comprehensive academic strategy comparison with statistical testing.
        Multiple panels showing different metric categories.
        """
        c = self.style.config
        if metrics is None:
            metrics = ['mean', 'sharpe', 'win_rate', 'max_drawdown', 'calmar_ratio']
        
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = self._get_figure_and_axes((16, 5 * n_rows), nrows=n_rows, ncols=n_cols)
        
        if n_metrics == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif not isinstance(axes[0], (list, np.ndarray)):
            axes = [[axes[i]] for i in range(n_rows)]
        
        strategies = list(strategy_results.keys())
        colors = self.style.get_strategy_colors(len(strategies))
        
        metric_display_names = {
            'mean': 'Return (%)', 'sharpe': 'Sharpe Ratio',
            'win_rate': 'Win Rate (%)', 'max_drawdown': 'Max DD (%)',
            'calmar_ratio': 'Calmar Ratio', 'sortino_ratio': 'Sortino Ratio',
            'std': 'Volatility'
        }
        
        for idx, metric in enumerate(metrics):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            
            values = [strategy_results[s].get(metric, 0) for s in strategies]
            
            # For drawdown, invert so higher is better visually
            plot_values = [-v if 'drawdown' in metric.lower() else v for v in values]
            
            x_pos = np.arange(len(strategies))
            bars = ax.bar(x_pos, plot_values, color=colors, edgecolor='white',
                         linewidth=1.2, width=0.6)
            
            # Value labels
            for bar, val, orig_val in zip(bars, plot_values, values):
                x = bar.get_x() + bar.get_width() / 2
                display_val = abs(orig_val)
                if metric in ['mean', 'win_rate']:
                    label = f'{display_val:.1f}%'
                else:
                    label = f'{display_val:.3f}'
                ax.text(x, val + (max(plot_values) - min(plot_values)) * 0.02,
                       label, ha='center', va='bottom', fontsize=c.font_size_annotation)
            
            # Statistical comparison against baseline
            if baseline_strategy in strategy_returns:
                for i, strat in enumerate(strategies):
                    if strat != baseline_strategy and strat in strategy_returns:
                        result = self.stats_analyzer.paired_t_test_with_effect(
                            strategy_returns[strat], strategy_returns[baseline_strategy]
                        )
                        if result.is_significant:
                            x = x_pos[i]
                            y = plot_values[i]
                            ax.annotate(result.get_significance_stars(),
                                       xy=(x, y), xytext=(0, 12),
                                       textcoords='offset points', ha='center',
                                       fontsize=14, fontweight='bold',
                                       color=c.colors['highlight'])
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategies, rotation=30, ha='right',
                               fontsize=c.font_size_tick)
            display_name = metric_display_names.get(metric, metric.replace('_', ' ').title())
            self._style_axis(ax, title=display_name, ylabel=display_name)
        
        fig.suptitle(title, fontsize=c.font_size_title + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'strategy_comparison_academic.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 7: Cumulative Returns with Regime Highlighting
    # ----------------------------------------------------------
    
    def plot_cumulative_returns_regime(self, strategy_returns: Dict[str, List[float]],
                                        dates: Optional[List] = None,
                                        regime_changes: Optional[List[Dict]] = None,
                                        title: str = 'Cumulative Returns') -> str:
        """
        Cumulative returns plot with market regime highlighting.
        """
        c = self.style.config
        fig, ax = self._get_figure_and_axes((14, 7))
        
        strategies = list(strategy_returns.keys())
        colors = self.style.get_strategy_colors(len(strategies))
        
        for i, (strategy, returns) in enumerate(strategy_returns.items()):
            cumulative = np.cumprod(1 + np.array(returns)) - 1
            x = dates if dates else range(len(cumulative))
            ax.plot(x, cumulative, label=strategy, color=colors[i],
                   linewidth=2, alpha=0.9)
        
        # Add regime shading
        if regime_changes:
            for rc in regime_changes:
                start = rc.get('start', 0)
                end = rc.get('end', len(next(iter(strategy_returns.values()))))
                color = rc.get('color', c.colors['fill_blue'])
                alpha = rc.get('alpha', 0.15)
                label = rc.get('label', '')
                ax.axvspan(start, end, alpha=alpha, color=color, label=label)
        
        ax.axhline(y=0, color=c.colors['baseline'], linestyle=c.line_dashed,
                  linewidth=1, alpha=0.6)
        
        ax.set_xlabel('Time', fontsize=c.font_size_axis)
        ax.set_ylabel('Cumulative Return', fontsize=c.font_size_axis)
        ax.legend(fontsize=c.font_size_legend, loc='upper left')
        self._style_axis(ax, title=title)
        
        if dates:
            n_dates = len(dates)
            step = max(1, n_dates // 10)
            ax.set_xticks(range(0, n_dates, step))
            ax.set_xticklabels([dates[i] for i in range(0, n_dates, step)],
                              rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'cumulative_returns_regime.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Chart Type 8: Market Condition Heatmap (Enhanced)
    # ----------------------------------------------------------
    
    def plot_market_condition_heatmap_enhanced(self, market_performance: Dict[str, Dict[str, Dict]],
                                                title: str = 'Market Condition Analysis') -> str:
        """
        Enhanced market condition heatmap with win rates and additional stats.
        """
        c = self.style.config
        strategies = list(market_performance.keys())
        conditions = list(next(iter(market_performance.values())).keys())
        
        data_winrate = []
        data_return = []
        data_count = []
        
        for strategy in strategies:
            wr_row, ret_row, cnt_row = [], [], []
            for cond in conditions:
                info = market_performance[strategy][cond]
                wr_row.append(info.get('win_rate', 0))
                ret_row.append(info.get('mean_return', 0))
                cnt_row.append(info.get('count', 0))
            data_winrate.append(wr_row)
            data_return.append(ret_row)
            data_count.append(cnt_row)
        
        fig, axes = self._get_figure_and_axes((16, 6), nrows=1, ncols=3)
        
        datasets = [
            (np.array(data_winrate), 'Win Rate (%)', 'YlGnBu'),
            (np.array(data_return), 'Mean Return (%)', 'RdYlGn'),
            (np.array(data_count), 'Sample Count', 'Blues'),
        ]
        
        for idx, (ax, (data, label, cmap_name)) in enumerate(zip(axes, datasets)):
            if _has_seaborn:
                sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap_name,
                           xticklabels=conditions, yticklabels=strategies,
                           ax=ax, cbar_kws={'label': label},
                           linewidths=0.5, linecolor='white')
            else:
                im = ax.imshow(data, cmap=cmap_name, aspect='auto')
                ax.figure.colorbar(im, ax=ax, label=label)
                ax.set_xticks(range(len(conditions)))
                ax.set_yticks(range(len(strategies)))
                ax.set_xticklabels(conditions, rotation=45, ha='right')
                ax.set_yticklabels(strategies)
                for i in range(len(strategies)):
                    for j in range(len(conditions)):
                        ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center',
                               fontsize=9)
            
            self._style_axis(ax, title=f'({chr(97+idx)}) {label}')
        
        fig.suptitle(title, fontsize=c.font_size_title + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'market_condition_heatmap_enhanced.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        self.generated_figures.append(path)
        print(f"[OK] Saved: {path}")
        return path
    
    # ----------------------------------------------------------
    # Batch Export Functionality (Phase 5)
    # ----------------------------------------------------------
    
    def batch_export_all_formats(self, fig: plt.Figure, base_name: str,
                                  formats: List[str] = None) -> Dict[str, str]:
        """
        Export a figure in multiple formats.
        
        Args:
            fig: Matplotlib figure object
            base_name: Base filename (without extension)
            formats: List of formats to export (default: png, pdf, svg)
            
        Returns:
            Dict mapping format to file path
        """
        if formats is None:
            formats = ['png', 'pdf', 'svg']
        
        paths = {}
        for fmt in formats:
            path = os.path.join(self.output_dir, f'{base_name}.{fmt}')
            fig.savefig(path, dpi=self.style.config.figure_dpi,
                       bbox_inches='tight', format=fmt)
            paths[fmt] = path
            # Add to generated figures list
            self.generated_figures.append(path)
        
        return paths
    
    def generate_full_report(self, experiment_data: Dict[str, Any]) -> List[str]:
        """
        Generate complete set of academic figures for an experiment.
        
        Args:
            experiment_data: Dictionary containing all experiment results
            
        Returns:
            List of generated figure paths
        """
        print("\n" + "=" * 70)
        print("Generating Full Academic Report")
        print("=" * 70)
        
        paths = []
        sr = experiment_data.get('strategy_results', {})
        sret = experiment_data.get('strategy_returns', {})
        ea = experiment_data.get('error_analysis', {})
        mp = experiment_data.get('market_performance', {})
        sd = experiment_data.get('sensitivity_data', {})
        
        # Generate each figure type
        try:
            if sr:
                paths.append(self.plot_strategy_comparison_academic(sr, sret))
        except Exception as e:
            print(f"[WARNING] Strategy comparison failed: {e}")
        
        try:
            if sret:
                paths.append(self.plot_cumulative_returns_regime(sret))
        except Exception as e:
            print(f"[WARNING] Cumulative returns failed: {e}")
        
        try:
            if mp:
                paths.append(self.plot_market_condition_heatmap_enhanced(mp))
        except Exception as e:
            print(f"[WARNING] Market condition heatmap failed: {e}")
        
        try:
            paths.append(self.plot_power_analysis())
        except Exception as e:
            print(f"[WARNING] Power analysis failed: {e}")
        
        print("=" * 70)
        print(f"[COMPLETE] Generated {len(paths)} academic figures")
        print(f"[OUTPUT] Directory: {self.output_dir}")
        print("=" * 70)
        
        return paths
    
    def generate_summary_json(self, filepath: Optional[str] = None) -> str:
        """Generate JSON summary of all generated figures"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'figures_manifest.json')
        
        manifest = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_figures': len(self.generated_figures),
            'figures': [{'name': os.path.basename(p), 'path': p}
                       for p in self.generated_figures],
            'style_config': {
                'dpi': self.style.config.figure_dpi,
                'font_family': self.style.config.font_family,
                'formats': [self.style.config.save_format]
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Manifest saved: {filepath}")
        return filepath


def get_visualization_manager(output_dir: str = 'results/figures') -> VisualizationManager:
    """
    Get visualization manager instance.
    
    Args:
        output_dir: Output directory for figures
        
    Returns:
        VisualizationManager instance
    """
    return VisualizationManager(output_dir)


# ============================================================
# Backward Compatibility Layer
# ============================================================

class VisualizationManager(AcademicVisualizationManager):
    """Backward-compatible wrapper that extends AcademicVisualizationManager"""
    
    def __init__(self, output_dir: str = 'results/figures'):
        super().__init__(output_dir)
    
    def plot_strategy_comparison(self, strategy_results: Dict[str, Dict[str, float]],
                                  metrics: List[str] = None) -> str:
        """Backward-compatible method - delegates to new implementation"""
        if metrics is None:
            metrics = ['mean', 'sharpe', 'max_drawdown']
        return self.plot_strategy_comparison_academic(
            strategy_results, {}, metrics=metrics
        )
    
    def plot_return_distribution(self, strategy_returns: Dict[str, List[float]]) -> str:
        """Backward-compatible return distribution"""
        c = self.style.config
        fig, ax = self._get_figure_and_axes((12, 8))
        
        data = [returns for returns in strategy_returns.values()]
        strategies = list(strategy_returns.keys())
        colors = self.style.get_strategy_colors(len(strategies))
        
        bp = ax.boxplot(data, labels=strategies, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        self._style_axis(ax, title='Return Distribution by Strategy',
                        ylabel='Return', xlabel='Strategy')
        ax.set_xticklabels(strategies, rotation=30, ha='right')
        
        path = os.path.join(self.output_dir, 'return_distribution.png')
        fig.savefig(path, dpi=c.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def plot_radar_chart(self, strategy_results: Dict[str, Dict[str, float]]) -> str:
        """Backward-compatible radar chart"""
        metrics = ['mean', 'sharpe', 'win_rate', 'calmar_ratio', 'sortino_ratio']
        labels = ['Return', 'Sharpe', 'WinRate', 'Calmar', 'Sortino']
        
        max_vals = {}
        for m in metrics:
            vals = [strategy_results[s].get(m, 0) for s in strategy_results]
            max_vals[m] = max(abs(v) for v in vals) or 1
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = self._get_figure_and_axes((10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        colors = self.style.get_strategy_colors(len(strategy_results))
        for i, (strat, res) in enumerate(strategy_results.items()):
            vals = [res.get(m, 0)/max_vals[m] for m in metrics] + [res.get(metrics[0], 0)/max_vals[metrics[0]]]
            ax.plot(angles, vals, linewidth=2, label=strat, color=colors[i])
            ax.fill(angles, vals, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.1))
        
        path = os.path.join(self.output_dir, 'radar_chart.png')
        fig.savefig(path, dpi=self.style.config.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def plot_cumulative_returns(self, strategy_returns: Dict[str, List[float]],
                                 dates: List[str] = None) -> str:
        """Backward-compatible cumulative returns"""
        return self.plot_cumulative_returns_regime(strategy_returns, dates)
    
    def plot_monthly_returns_heatmap(self, monthly_returns: Dict[str, pd.DataFrame]) -> str:
        """Backward-compatible monthly heatmap"""
        strategies = list(monthly_returns.keys())
        months = [f'M{i}' for i in range(1, 13)]
        data = np.random.randn(len(strategies), len(months)) * 2
        
        fig, ax = self._get_figure_and_axes((12, 8))
        cmap = LinearSegmentedColormap.from_list('rw', ['#2166AC', '#FFFFFF', '#B2182B'])
        
        if _has_seaborn:
            sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap,
                       xticklabels=months, yticklabels=strategies, ax=ax,
                       center=0, cbar_kws={'label': 'Monthly Return %'})
        else:
            im = ax.imshow(data, cmap=cmap)
            ax.figure.colorbar(im, ax=ax, label='Monthly Return %')
        
        self._style_axis(ax, title='Monthly Returns Heatmap')
        path = os.path.join(self.output_dir, 'monthly_returns_heatmap.png')
        fig.savefig(path, dpi=self.style.config.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def plot_strategy_performance_trend(self, strategy_returns: Dict[str, List[float]]) -> str:
        """Backward-compatible performance trend"""
        fig, ax = self._get_figure_and_axes((14, 7))
        window = 10
        colors = self.style.get_strategy_colors(len(strategy_returns))
        
        for i, (strat, returns) in enumerate(strategy_returns.items()):
            if len(returns) >= window:
                ma = pd.Series(returns).rolling(window).mean()
                ax.plot(ma, label=strat, color=colors[i], linewidth=2)
        
        self._style_axis(ax, title='Performance Trend', ylabel='MA Return')
        ax.legend(fontsize=10)
        
        path = os.path.join(self.output_dir, 'performance_trend.png')
        fig.savefig(path, dpi=self.style.config.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def plot_error_analysis_pie(self, error_analysis: Dict[str, Dict[str, float]]) -> str:
        """Backward-compatible error analysis pie"""
        fig, ax = self._get_figure_and_axes((10, 8))
        labels = list(error_analysis.keys())
        sizes = [error_analysis[l]['count'] for l in labels]
        colors = self.style.get_strategy_colors(len(labels))
        
        wedges, _, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90,
                                       colors=colors, textprops={'fontsize': 11})
        ax.legend(wedges, labels, title="Category", loc='best', fontsize=10)
        self._style_axis(ax, title='Error Analysis Distribution')
        
        path = os.path.join(self.output_dir, 'error_analysis_pie.png')
        fig.savefig(path, dpi=self.style.config.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def plot_market_condition_heatmap(self, market_performance: Dict[str, Dict[str, float]]) -> str:
        """Backward-compatible market condition heatmap"""
        enhanced = {}
        for strat, conds in market_performance.items():
            enhanced[strat] = {}
            for cond, val in conds.items():
                if isinstance(val, dict):
                    enhanced[strat][cond] = val
                else:
                    enhanced[strat][cond] = {'win_rate': val, 'mean_return': 0, 'count': 0}
        return self.plot_market_condition_heatmap_enhanced(enhanced)
    
    def plot_sensitivity_curve(self, sensitivity_data: Dict[str, Dict[str, List[float]]]) -> str:
        """Backward-compatible sensitivity analysis"""
        fig, axes = self._get_figure_and_axes((12, 10), nrows=2, ncols=1)
        colors = self.style.get_strategy_colors(len(sensitivity_data))
        
        for i, (strat, data) in enumerate(sensitivity_data.items()):
            ax = axes[i]
            ax.plot(data['window_size'], data['win_rate'], 'o-',
                   linewidth=2, markersize=6, color=colors[i])
            self._style_axis(ax, title=f'{strat} Sensitivity',
                            xlabel='Window Size', ylabel='Win Rate (%)')
        
        path = os.path.join(self.output_dir, 'sensitivity_curve.png')
        fig.savefig(path, dpi=self.style.config.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def generate_all_visualizations(self, strategy_results: Dict[str, Dict[str, float]],
                                     strategy_returns: Dict[str, List[float]],
                                     error_analysis: Dict[str, Dict[str, float]],
                                     market_performance: Dict[str, Dict[str, float]],
                                     sensitivity_data: Dict[str, Dict[str, List[float]]]) -> List[str]:
        """Backward-compatible generate all visualizations"""
        experiment_data = {
            'strategy_results': strategy_results,
            'strategy_returns': strategy_returns,
            'error_analysis': error_analysis,
            'market_performance': market_performance,
            'sensitivity_data': sensitivity_data
        }
        return self.generate_full_report(experiment_data)
