#!/usr/bin/env python3
"""
Plot Generator for Benchmark Suite

Generates publication-ready plots:
- Violin plots for timing distributions (Seaborn)
- Bar charts for mean comparisons
- Combined figures for paper columns

Style inspired by academic paper standards.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ═══════════════════════════════════════════════════════════════════════════
# Plot Configuration for Paper
# ═══════════════════════════════════════════════════════════════════════════

# IEEE/ACM single column width: ~3.5 inches
# Double column width: ~7 inches
COLUMN_WIDTH = 3.5  # inches
DOUBLE_COLUMN_WIDTH = 7.0

# Color palette (colorblind-friendly, inspired by your paper)
COLORS = {
    "python": "#5DA5DA",   # Blue (like "Baseline" in your image)
    "cpp": "#FAA43A",      # Orange (like "Human Opt")
    "rust": "#60BD68",     # Green (like "LLM Opt")
}

# Alternative muted palette
COLORS_MUTED = {
    "python": "#4878A8",   # Muted blue
    "cpp": "#E8983E",      # Muted orange
    "rust": "#6AAF6A",     # Muted green
}

DISPLAY_NAMES = {
    "python": "Python Native",
    "cpp": "C++ WASM",
    "rust": "Rust WASM",
}

# Short names for tight spaces
SHORT_NAMES = {
    "python": "Python",
    "cpp": "C++ WASM",
    "rust": "Rust WASM",
}

# Font sizes for paper
FONT_SIZES = {
    "title": 11,
    "label": 10,
    "tick": 9,
    "legend": 9,
    "annotation": 8,
    "value": 8,
}


class PlotGenerator:
    """Generate publication-ready plots"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global style
        self._setup_style()
        
    def _setup_style(self):
        """Setup matplotlib/seaborn style"""
        if HAS_SEABORN:
            sns.set_style("whitegrid", {
                'grid.linestyle': '--',
                'grid.alpha': 0.6,
                'axes.edgecolor': '#333333',
                'axes.linewidth': 1.0,
            })
            sns.set_context("paper", font_scale=1.1)
        
        plt.rcParams.update({
            'font.size': FONT_SIZES['tick'],
            'axes.titlesize': FONT_SIZES['title'],
            'axes.labelsize': FONT_SIZES['label'],
            'xtick.labelsize': FONT_SIZES['tick'],
            'ytick.labelsize': FONT_SIZES['tick'],
            'legend.fontsize': FONT_SIZES['legend'],
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.family': 'serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
    def generate_all(self, report):
        """Generate all plots"""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib not available, skipping plots")
            return
            
        print("\n📊 Generating plots...")
        
        # Individual plots
        self.plot_violin_attestation(report)
        self.plot_violin_training(report)
        self.plot_violin_inference(report)
        self.plot_bar_comparison(report)
        self.plot_bar_total_time(report)
        
        # Combined figure for paper
        self.plot_combined_figure(report)
        
        print(f"   Plots saved to: {self.output_dir}")
        
    def _prepare_data(self, report, metric: str) -> tuple:
        """Prepare data for plotting"""
        benchmarks = ["python", "cpp", "rust"]
        data = []
        labels = []
        colors = []
        names = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                metric_data = getattr(stats, metric, None)
                if metric_data and metric_data.values:
                    data.append(metric_data.values)
                    labels.append(DISPLAY_NAMES.get(name, name))
                    colors.append(COLORS.get(name, "#888888"))
                    names.append(name)
                    
        return data, labels, colors, names
    
    def _create_seaborn_violin_data(self, report, metric: str):
        """Prepare data in long format for seaborn"""
        import pandas as pd
        
        benchmarks = ["python", "cpp", "rust"]
        rows = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                metric_data = getattr(stats, metric, None)
                if metric_data and metric_data.values:
                    for val in metric_data.values:
                        rows.append({
                            'Implementation': DISPLAY_NAMES.get(name, name),
                            'Time (ms)': val,
                            'name': name
                        })
        
        return pd.DataFrame(rows) if rows else None
    
    def plot_violin_attestation(self, report):
        """Violin plot for attestation times using Seaborn"""
        if not HAS_SEABORN:
            return self._plot_violin_matplotlib(report, "attestation", "TEE Attestation Time")
            
        import pandas as pd
        
        df = self._create_seaborn_violin_data(report, "attestation")
        if df is None or df.empty:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.5, 3))
        
        # Create violin plot with seaborn
        palette = [COLORS[name] for name in ["python", "cpp", "rust"] 
                   if name in df['name'].unique()]
        
        sns.violinplot(
            data=df, 
            x='Implementation', 
            y='Time (ms)',
            palette=palette,
            inner='box',  # Show box plot inside
            linewidth=1,
            ax=ax
        )
        
        # Add mean annotations
        for i, name in enumerate(["python", "cpp", "rust"]):
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                if stats.attestation and stats.attestation.values:
                    mean_val = stats.attestation.mean
                    std_val = stats.attestation.std
                    ax.annotate(
                        f'μ={mean_val:.0f}\nσ={std_val:.0f}',
                        xy=(i, max(stats.attestation.values)),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=FONT_SIZES['annotation'],
                        fontweight='bold'
                    )
        
        ax.set_title('TEE Attestation Time Distribution', fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_attestation.pdf')
        plt.savefig(self.output_dir / 'violin_attestation.png')
        plt.close()
        
    def plot_violin_training(self, report):
        """Violin plot for training times using Seaborn"""
        if not HAS_SEABORN:
            return self._plot_violin_matplotlib(report, "training", "Training Time")
            
        import pandas as pd
        
        df = self._create_seaborn_violin_data(report, "training")
        if df is None or df.empty:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.5, 3))
        
        palette = [COLORS[name] for name in ["python", "cpp", "rust"] 
                   if name in df['name'].unique()]
        
        sns.violinplot(
            data=df,
            x='Implementation',
            y='Time (ms)',
            palette=palette,
            inner='box',
            linewidth=1,
            ax=ax
        )
        
        # Add mean annotations
        for i, name in enumerate(["python", "cpp", "rust"]):
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                if stats.training and stats.training.values:
                    mean_val = stats.training.mean
                    std_val = stats.training.std
                    ax.annotate(
                        f'μ={mean_val:.0f}\nσ={std_val:.0f}',
                        xy=(i, max(stats.training.values)),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=FONT_SIZES['annotation'],
                        fontweight='bold'
                    )
        
        ax.set_title('Training Time Distribution (200 trees)', fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_training.pdf')
        plt.savefig(self.output_dir / 'violin_training.png')
        plt.close()
        
    def plot_violin_inference(self, report):
        """Violin plot for inference times using Seaborn"""
        if not HAS_SEABORN:
            return self._plot_violin_matplotlib(report, "inference", "Inference Time")
            
        import pandas as pd
        
        df = self._create_seaborn_violin_data(report, "inference")
        if df is None or df.empty:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.5, 3))
        
        palette = [COLORS[name] for name in ["python", "cpp", "rust"] 
                   if name in df['name'].unique()]
        
        sns.violinplot(
            data=df,
            x='Implementation',
            y='Time (ms)',
            palette=palette,
            inner='box',
            linewidth=1,
            ax=ax
        )
        
        # Add mean annotations
        for i, name in enumerate(["python", "cpp", "rust"]):
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                if stats.inference and stats.inference.values:
                    mean_val = stats.inference.mean
                    std_val = stats.inference.std
                    ax.annotate(
                        f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                        xy=(i, max(stats.inference.values)),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=FONT_SIZES['annotation'],
                        fontweight='bold'
                    )
        
        ax.set_title('Inference Time Distribution (89 samples)', fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_inference.pdf')
        plt.savefig(self.output_dir / 'violin_inference.png')
        plt.close()

    def _plot_violin_matplotlib(self, report, metric: str, title: str):
        """Fallback violin plot using matplotlib"""
        data, labels, colors, names = self._prepare_data(report, metric)
        
        if not data:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.5, 3))
        
        parts = ax.violinplot(data, positions=range(len(data)),
                              showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Time (ms)')
        ax.set_title(title, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'violin_{metric}.pdf')
        plt.savefig(self.output_dir / f'violin_{metric}.png')
        plt.close()
        
    def plot_bar_comparison(self, report):
        """Bar chart comparing all metrics - style like your paper"""
        benchmarks = ["python", "cpp", "rust"]
        metrics = ["attestation", "training", "inference"]
        metric_labels = ["Attestation", "Training", "Inference"]
        
        # Prepare data
        means = {m: [] for m in metrics}
        stds = {m: [] for m in metrics}
        labels = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                labels.append(SHORT_NAMES.get(name, name))
                for m in metrics:
                    metric_data = getattr(stats, m, None)
                    if metric_data:
                        means[m].append(metric_data.mean)
                        stds[m].append(metric_data.std)
                    else:
                        means[m].append(0)
                        stds[m].append(0)
        
        if not labels:
            return
        
        # Taller figure to accommodate vertical labels and legend
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, 4.5))
        
        x = np.arange(len(labels))
        width = 0.25
        
        # Colors matching your paper style
        metric_colors = ['#5DA5DA', '#FAA43A', '#60BD68']  # Blue, Orange, Green
        
        bars_list = []
        for i, (m, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means[m], width,
                         label=label, color=color,
                         yerr=stds[m], capsize=3, 
                         edgecolor='#333333', linewidth=0.5,
                         error_kw={'linewidth': 1, 'capthick': 1})
            bars_list.append(bars)
            
            # Add vertical value labels on top of each bar
            for j, (bar, val) in enumerate(zip(bars, means[m])):
                height = bar.get_height() + stds[m][j]
                # Format based on magnitude
                if val >= 1000:
                    label_text = f'{val:.0f}'
                elif val >= 10:
                    label_text = f'{val:.1f}'
                else:
                    label_text = f'{val:.2f}'
                    
                ax.annotate(label_text,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=FONT_SIZES['value'],
                           rotation=90,
                           fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Benchmark Comparison', fontweight='bold', pad=10)
        
        # Log scale for better visualization
        ax.set_yscale('log')
        
        # Legend outside/above to avoid overlap
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                 ncol=3, frameon=True, fancybox=True, shadow=False,
                 edgecolor='#cccccc')
        
        # Add some padding at top for labels
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 2.5)
        
        # Grid only on y-axis
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bar_comparison.pdf')
        plt.savefig(self.output_dir / 'bar_comparison.png')
        plt.close()
        
    def plot_bar_total_time(self, report):
        """Stacked bar chart for total time breakdown"""
        benchmarks = ["python", "cpp", "rust"]
        
        labels = []
        attestation = []
        training = []
        inference = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                labels.append(SHORT_NAMES.get(name, name))
                attestation.append(stats.attestation.mean if stats.attestation else 0)
                training.append(stats.training.mean if stats.training else 0)
                inference.append(stats.inference.mean if stats.inference else 0)
        
        if not labels:
            return
            
        # Taller figure
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.5, 4))
        
        x = np.arange(len(labels))
        width = 0.5
        
        # Stack colors
        colors = ['#95a5a6', '#3498db', '#2ecc71']  # Grey, Blue, Green
        
        # Create stacked bars
        p1 = ax.bar(x, attestation, width, label='Attestation', 
                   color=colors[0], edgecolor='#333333', linewidth=0.5)
        p2 = ax.bar(x, training, width, bottom=attestation, 
                   label='Training', color=colors[1], edgecolor='#333333', linewidth=0.5)
        
        bottom2 = [a + t for a, t in zip(attestation, training)]
        p3 = ax.bar(x, inference, width, bottom=bottom2, 
                   label='Inference', color=colors[2], edgecolor='#333333', linewidth=0.5)
        
        # Add component labels inside bars (if large enough)
        for i in range(len(labels)):
            # Attestation label
            if attestation[i] > 500:
                ax.annotate(f'{attestation[i]:.0f}',
                           xy=(i, attestation[i] / 2),
                           ha='center', va='center',
                           fontsize=FONT_SIZES['value'],
                           color='white', fontweight='bold')
            
            # Training label
            if training[i] > 500:
                ax.annotate(f'{training[i]:.0f}',
                           xy=(i, attestation[i] + training[i] / 2),
                           ha='center', va='center',
                           fontsize=FONT_SIZES['value'],
                           color='white', fontweight='bold')
            
            # Inference is typically small, skip internal label
        
        # Total time labels on top
        totals = [a + t + i for a, t, i in zip(attestation, training, inference)]
        for i, total in enumerate(totals):
            ax.annotate(f'{total:.0f} ms',
                       xy=(i, total),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=FONT_SIZES['annotation'],
                       fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Total Execution Time Breakdown', fontweight='bold', pad=10)
        
        # Legend below
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=3, frameon=True, fancybox=True,
                 edgecolor='#cccccc')
        
        # Add padding at top
        ymax = max(totals)
        ax.set_ylim(0, ymax * 1.15)
        
        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bar_total_time.pdf')
        plt.savefig(self.output_dir / 'bar_total_time.png')
        plt.close()
        
    def plot_combined_figure(self, report):
        """Combined figure for paper (single column) with Seaborn violins"""
        benchmarks = ["python", "cpp", "rust"]
        
        if HAS_SEABORN:
            import pandas as pd
        
        fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH + 0.5, 8.5))
        
        # --- Subplot 1: Violin plot for training ---
        ax = axes[0]
        
        if HAS_SEABORN:
            df = self._create_seaborn_violin_data(report, "training")
            if df is not None and not df.empty:
                palette = [COLORS[name] for name in ["python", "cpp", "rust"] 
                          if name in df['name'].unique()]
                
                sns.violinplot(
                    data=df,
                    x='Implementation',
                    y='Time (ms)',
                    palette=palette,
                    inner='box',
                    linewidth=1,
                    ax=ax
                )
                
                # Add annotations
                for i, name in enumerate(["python", "cpp", "rust"]):
                    if name in report.benchmarks:
                        stats = report.benchmarks[name]
                        if stats.training and stats.training.values:
                            mean_val = stats.training.mean
                            std_val = stats.training.std
                            ymax = max(stats.training.values)
                            ax.annotate(
                                f'μ={mean_val:.0f}\nσ={std_val:.0f}',
                                xy=(i, ymax),
                                xytext=(0, 8),
                                textcoords='offset points',
                                ha='center',
                                fontsize=FONT_SIZES['annotation'] - 1,
                                fontweight='bold'
                            )
        
        ax.set_title('(a) Training Time Distribution', fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Time (ms)')
        
        # --- Subplot 2: Violin plot for inference ---
        ax = axes[1]
        
        if HAS_SEABORN:
            df = self._create_seaborn_violin_data(report, "inference")
            if df is not None and not df.empty:
                palette = [COLORS[name] for name in ["python", "cpp", "rust"] 
                          if name in df['name'].unique()]
                
                sns.violinplot(
                    data=df,
                    x='Implementation',
                    y='Time (ms)',
                    palette=palette,
                    inner='box',
                    linewidth=1,
                    ax=ax
                )
                
                for i, name in enumerate(["python", "cpp", "rust"]):
                    if name in report.benchmarks:
                        stats = report.benchmarks[name]
                        if stats.inference and stats.inference.values:
                            mean_val = stats.inference.mean
                            std_val = stats.inference.std
                            ymax = max(stats.inference.values)
                            ax.annotate(
                                f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                                xy=(i, ymax),
                                xytext=(0, 8),
                                textcoords='offset points',
                                ha='center',
                                fontsize=FONT_SIZES['annotation'] - 1,
                                fontweight='bold'
                            )
        
        ax.set_title('(b) Inference Time Distribution', fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Time (ms)')
        
        # --- Subplot 3: Stacked bar for total time ---
        ax = axes[2]
        
        labels_bar = []
        attestation = []
        training = []
        inference = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                labels_bar.append(SHORT_NAMES.get(name, name))
                attestation.append(stats.attestation.mean if stats.attestation else 0)
                training.append(stats.training.mean if stats.training else 0)
                inference.append(stats.inference.mean if stats.inference else 0)
        
        if labels_bar:
            x = np.arange(len(labels_bar))
            width = 0.5
            
            colors = ['#95a5a6', '#3498db', '#2ecc71']
            
            p1 = ax.bar(x, attestation, width, label='Attestation', 
                       color=colors[0], edgecolor='#333333', linewidth=0.5)
            p2 = ax.bar(x, training, width, bottom=attestation, 
                       label='Training', color=colors[1], edgecolor='#333333', linewidth=0.5)
            
            bottom2 = [a + t for a, t in zip(attestation, training)]
            p3 = ax.bar(x, inference, width, bottom=bottom2, 
                       label='Inference', color=colors[2], edgecolor='#333333', linewidth=0.5)
            
            totals = [a + t + i for a, t, i in zip(attestation, training, inference)]
            for i, total in enumerate(totals):
                ax.annotate(f'{total:.0f} ms',
                           xy=(i, total),
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center',
                           fontsize=FONT_SIZES['annotation'],
                           fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels_bar, fontweight='bold')
            ax.set_ylabel('Time (ms)')
            ax.set_title('(c) Total Execution Time Breakdown', fontweight='bold', pad=10)
            ax.legend(loc='upper right', framealpha=0.95, fontsize=7,
                     edgecolor='#cccccc')
            
            ymax = max(totals)
            ax.set_ylim(0, ymax * 1.12)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.xaxis.grid(False)
        
        plt.tight_layout(h_pad=2.0)
        plt.savefig(self.output_dir / 'combined_figure.pdf')
        plt.savefig(self.output_dir / 'combined_figure.png')
        plt.close()
        
        print(f"   📈 Combined figure: {self.output_dir / 'combined_figure.pdf'}")

    def plot_bar_grouped_paper_style(self, report):
        """
        Bar chart in paper style (like your uploaded image)
        Grouped bars with values on top
        """
        benchmarks = ["python", "cpp", "rust"]
        
        # Prepare data
        labels = []
        attestation = []
        training = []
        inference = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                labels.append(SHORT_NAMES.get(name, name))
                attestation.append(stats.attestation.mean if stats.attestation else 0)
                training.append(stats.training.mean if stats.training else 0)
                inference.append(stats.inference.mean if stats.inference else 0)
        
        if not labels:
            return
        
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1, 4))
        
        x = np.arange(len(labels))
        width = 0.25
        
        # Colors like your paper
        colors = ['#5DA5DA', '#FAA43A', '#60BD68']
        
        # Create grouped bars
        bars1 = ax.bar(x - width, attestation, width, label='Attestation',
                      color=colors[0], edgecolor='#333333', linewidth=0.5)
        bars2 = ax.bar(x, training, width, label='Training',
                      color=colors[1], edgecolor='#333333', linewidth=0.5)
        bars3 = ax.bar(x + width, inference, width, label='Inference',
                      color=colors[2], edgecolor='#333333', linewidth=0.5)
        
        # Add value labels on top (vertical)
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if val >= 100:
                    label = f'{val:.0f}'
                elif val >= 1:
                    label = f'{val:.1f}'
                else:
                    label = f'{val:.2f}'
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=FONT_SIZES['value'],
                           rotation=90,
                           fontweight='bold')
        
        add_labels(bars1, attestation)
        add_labels(bars2, training)
        add_labels(bars3, inference)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Benchmark Algorithm', fontweight='bold')
        
        # Log scale
        ax.set_yscale('log')
        
        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                 ncol=3, frameon=True)
        
        # Padding
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 3)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bar_paper_style.pdf')
        plt.savefig(self.output_dir / 'bar_paper_style.png')
        plt.close()
