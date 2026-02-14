"""
Visualization utilities for change point detection and risk metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging

from src.change_point import ChangePoint
from src.risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Creates professional visualizations for financial analysis.
    """

    def __init__(self, style: str = 'finance'):
        """
        Initialize visualizer with style settings.

        Args:
            style: Visualization style ('finance' or 'presentation')
        """
        self.style = style
        self._configure_styles()

    def _configure_styles(self):
        """Configure matplotlib styles for professional appearance."""
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['lines.linewidth'] = 1.5

        if self.style == 'finance':
            plt.rcParams['axes.facecolor'] = '#f8f9fa'
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['grid.color'] = '#dee2e6'
            plt.rcParams['grid.alpha'] = 0.3

    def plot_price_with_change_points(
        self,
        prices: pd.Series,
        change_points: List[ChangePoint],
        title: str = "Brent Oil Price with Structural Breaks",
        save_path: Optional[Path] = None,
        show_confidence: bool = True
    ) -> plt.Figure:
        """
        Plot price series with detected change points.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price line
        ax.plot(prices.index, prices.values, linewidth=1.5, color='#1f77b4',
                label='Brent Oil Price', alpha=0.8)

        # Color code by significance
        significance_colors = {
            'High': '#d62728',   # Red
            'Medium': '#ff7f0e', # Orange
            'Low': '#2ca02c',    # Green
            'Minor': '#7f7f7f'   # Gray
        }

        # Plot change points
        for cp in change_points:
            color = significance_colors.get(cp.significance, '#000000')
            marker = '^' if cp.direction == 'Increase' else 'v'

            ax.scatter(cp.date, cp.price_after, color=color, s=100,
                       marker=marker, zorder=5, alpha=0.8)

            if show_confidence:
                ax.annotate(
                    f'{cp.change_pct:.1f}%\n({cp.confidence:.2f})',
                    xy=(cp.date, cp.price_after),
                    xytext=(10, 10 if cp.direction == 'Increase' else -20),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price (USD per barrel)', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)

        # Add summary text
        summary = self._create_change_point_summary(change_points)
        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")

        return fig

    def plot_risk_dashboard(
        self,
        prices: pd.Series,
        risk_metrics: RiskMetrics,
        change_points: Optional[List[ChangePoint]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive risk dashboard.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main price chart
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_price_chart(ax_main, prices, change_points)

        # Rolling volatility
        ax_vol = fig.add_subplot(gs[1, 0])
        self._plot_rolling_volatility(ax_vol, risk_metrics.rolling_volatility)

        # Drawdown
        ax_dd = fig.add_subplot(gs[1, 1])
        self._plot_drawdown(ax_dd, risk_metrics.drawdown_series)

        # VaR comparison
        ax_var = fig.add_subplot(gs[1, 2])
        self._plot_var_comparison(ax_var, risk_metrics)

        # Returns distribution
        ax_returns = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax_returns, prices)

        # Risk ratios
        ax_ratios = fig.add_subplot(gs[2, 1])
        self._plot_risk_ratios(ax_ratios, risk_metrics)

        # Stress test results
        ax_stress = fig.add_subplot(gs[2, 2])
        self._plot_stress_tests(ax_stress, risk_metrics.stress_test_results)

        # Add title
        fig.suptitle('Brent Oil Risk Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dashboard to {save_path}")

        return fig

    # ---------------- Helper plotting functions ----------------

    def _plot_price_chart(self, ax, prices, change_points):
        ax.plot(prices.index, prices.values, linewidth=1.5, color='#1f77b4', alpha=0.8)
        if change_points:
            for cp in change_points:
                ax.axvline(x=cp.date, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.scatter(cp.date, cp.price_after, color='red', s=50, zorder=5)
        ax.set_title('Brent Oil Price History', fontweight='bold')
        ax.set_ylabel('Price (USD)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))

    def _plot_rolling_volatility(self, ax, rolling_vol):
        ax.plot(rolling_vol.index, rolling_vol.values, color='#ff7f0e', linewidth=1.5)
        ax.axhline(y=rolling_vol.mean(), color='red', linestyle='--', alpha=0.5,
                   label=f'Mean: {rolling_vol.mean():.2%}')
        ax.set_title('Rolling Volatility (30-day)', fontweight='bold')
        ax.set_ylabel('Annualized Vol')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, ax, drawdown):
        ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                        color='#d62728', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values * 100, color='#d62728', linewidth=1)
        ax.set_title('Drawdown Analysis', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.set_ylim(min(drawdown * 100) * 1.1, 5)
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_var_comparison(self, ax, risk_metrics):
        conf_levels = list(risk_metrics.var_historical.keys())
        hist_var = [risk_metrics.var_historical[c] * 100 for c in conf_levels]
        param_var = [risk_metrics.var_parametric[c] * 100 for c in conf_levels]

        x = np.arange(len(conf_levels))
        width = 0.35

        ax.bar(x - width / 2, hist_var, width, label='Historical', color='#1f77b4', alpha=0.8)
        ax.bar(x + width / 2, param_var, width, label='Parametric', color='#ff7f0e', alpha=0.8)

        ax.set_title('Value at Risk (VaR) Comparison', fontweight='bold')
        ax.set_ylabel('VaR (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(c * 100)}%' for c in conf_levels])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_returns_distribution(self, ax, prices):
        returns = prices.pct_change().dropna() * 100
        ax.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
        ax.axvline(x=returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
        ax.axvline(x=returns.median(), color='blue', linestyle='--', label=f'Median: {returns.median():.2f}%')
        ax.set_title('Daily Returns Distribution', fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_risk_ratios(self, ax, risk_metrics):
        ratios = ['Sharpe', 'Sortino', 'Calmar']
        values = [
            risk_metrics.sharpe_ratio,
            risk_metrics.sortino_ratio,
            risk_metrics.calmar_ratio
        ]
        colors = ['#1f77b4' if v > 0 else '#d62728' for v in values]

        bars = ax.bar(ratios, values, color=colors, alpha=0.8)
        for bar, v in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title('Risk-Adjusted Return Ratios', fontweight='bold')
        ax.set_ylabel('Ratio')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_stress_tests(self, ax, stress_results):
        scenarios = list(stress_results.keys())
        impacts = list(stress_results.values())
        colors = ['#d62728' if i < 0 else '#2ca02c' for i in impacts]
        y_pos = np.arange(len(scenarios))
        bars = ax.barh(y_pos, impacts, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios], fontsize=8)
        ax.set_title('Stress Test Scenarios', fontweight='bold')
        ax.set_xlabel('Price Impact (%)')

        for bar, impact in zip(bars, impacts):
            width = bar.get_width()
            ax.text(width + (0.5 if width >= 0 else -1), bar.get_y() + bar.get_height() / 2.,
                    f'{impact:.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=8)

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    # ---------------- Change point impact analysis ----------------

    def _create_change_point_summary(self, change_points: List[ChangePoint]) -> str:
        if not change_points:
            return "No significant change points detected"

        total = len(change_points)
        high = sum(1 for cp in change_points if cp.significance == 'High')
        medium = sum(1 for cp in change_points if cp.significance == 'Medium')
        avg_change = np.mean([abs(cp.change_pct) for cp in change_points])
        avg_confidence = np.mean([cp.confidence for cp in change_points])

        return (f"Change Points Summary:\n"
                f"Total: {total}\n"
                f"High significance: {high}\n"
                f"Medium significance: {medium}\n"
                f"Avg change: {avg_change:.1f}%\n"
                f"Avg confidence: {avg_confidence:.2f}")

    def plot_change_point_impact(
        self,
        change_points: List[ChangePoint],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Magnitude
        ax1 = axes[0, 0]
        dates = [cp.date for cp in change_points]
        changes = [abs(cp.change_pct) for cp in change_points]
        ax1.scatter(dates, changes, c=changes, cmap='RdYlGn_r', s=100, alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Change Magnitude (%)')
        ax1.set_title('Change Point Magnitudes Over Time')
        ax1.grid(True, alpha=0.3)

        # Confidence
        ax2 = axes[0, 1]
        confidences = [cp.confidence for cp in change_points]
        ax2.hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='#1f77b4')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Change Point Confidence Distribution')
        ax2.grid(True, alpha=0.3)

        # Direction pie chart
        ax3 = axes[1, 0]
        increases = sum(1 for cp in change_points if cp.direction == 'Increase')
        decreases = len(change_points) - increases
        ax3.pie([increases, decreases], labels=['Increases', 'Decreases'],
                autopct='%1.1f%%', colors=['#2ca02c', '#d62728'], startangle=90)
        ax3.set_title('Change Direction Distribution')

        # Volatility impact
        ax4 = axes[1, 1]
        vol_ratios = [cp.volatility_after / cp.volatility_before
                      for cp in change_points if cp.volatility_before > 0]
        if vol_ratios:
            ax4.bar(range(len(vol_ratios)), vol_ratios, alpha=0.7, color='#ff7f0e')
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Change Point Index')
            ax4.set_ylabel('Volatility Ratio (After/Before)')
            ax4.set_title('Impact on Volatility')
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Change Point Impact Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved change point impact figure to {save_path}")

        return fig
