#!/usr/bin/env python3
"""
Plot Generator for Benchmark Suite

Generates publication-ready plots:
- Violin plots for timing distributions
- Bar charts for mean comparisons
- Combined figures for paper columns
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
    from matplotlib.ticker import FuncFormatter
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

# Color palette (colorblind-friendly)
COLORS = {
    "python": "#2ecc71",   # Green
    "cpp": "#3498db",      # Blue  
    "rust": "#e74c3c",     # Red/Orange
}

DISPLAY_NAMES = {
    "python": "Python Native",
    "cpp": "C++ WASM",
    "rust": "Rust WASM",
}

# Font sizes for paper
FONT_SIZES = {
    "title": 10,
    "label": 9,
    "tick": 8,
    "legend": 8,
    "annotation": 7,
}


class PlotGenerator:
    """Generate publication-ready plots"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_context("paper")
        
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
            'font.family': 'serif',
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
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                metric_data = getattr(stats, metric, None)
                if metric_data and metric_data.values:
                    data.append(metric_data.values)
                    labels.append(DISPLAY_NAMES.get(name, name))
                    colors.append(COLORS.get(name, "#888888"))
                    
        return data, labels, colors
    
    def plot_violin_attestation(self, report):
        """Violin plot for attestation times"""
        data, labels, colors = self._prepare_data(report, "attestation")
        
        if not data:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
        
        parts = ax.violinplot(data, positions=range(len(data)), 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            
        # Style the lines
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_edgecolor('#333333')
                parts[partname].set_linewidth(1)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Time (ms)')
        ax.set_title('TEE Attestation Time')
        
        # Add mean values as annotations
        for i, d in enumerate(data):
            mean_val = np.mean(d)
            ax.annotate(f'{mean_val:.1f}', 
                       xy=(i, mean_val), 
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=FONT_SIZES['annotation'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_attestation.pdf')
        plt.savefig(self.output_dir / 'violin_attestation.png')
        plt.close()
        
    def plot_violin_training(self, report):
        """Violin plot for training times"""
        data, labels, colors = self._prepare_data(report, "training")
        
        if not data:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
        
        parts = ax.violinplot(data, positions=range(len(data)),
                              showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_edgecolor('#333333')
                parts[partname].set_linewidth(1)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Training Time (200 trees)')
        
        for i, d in enumerate(data):
            mean_val = np.mean(d)
            ax.annotate(f'{mean_val:.0f}', 
                       xy=(i, mean_val),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=FONT_SIZES['annotation'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_training.pdf')
        plt.savefig(self.output_dir / 'violin_training.png')
        plt.close()
        
    def plot_violin_inference(self, report):
        """Violin plot for inference times"""
        data, labels, colors = self._prepare_data(report, "inference")
        
        if not data:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
        
        parts = ax.violinplot(data, positions=range(len(data)),
                              showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_edgecolor('#333333')
                parts[partname].set_linewidth(1)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Inference Time (89 samples)')
        
        for i, d in enumerate(data):
            mean_val = np.mean(d)
            ax.annotate(f'{mean_val:.2f}', 
                       xy=(i, mean_val),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=FONT_SIZES['annotation'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_inference.pdf')
        plt.savefig(self.output_dir / 'violin_inference.png')
        plt.close()
        
    def plot_bar_comparison(self, report):
        """Bar chart comparing all metrics"""
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
                labels.append(DISPLAY_NAMES.get(name, name))
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
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3))
        
        x = np.arange(len(labels))
        width = 0.25
        
        metric_colors = ['#95a5a6', '#3498db', '#2ecc71']
        
        for i, (m, label) in enumerate(zip(metrics, metric_labels)):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means[m], width, 
                         label=label, color=metric_colors[i],
                         yerr=stds[m], capsize=3, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Time (ms)')
        ax.set_title('Benchmark Comparison')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_yscale('log')
        
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
                labels.append(DISPLAY_NAMES.get(name, name))
                attestation.append(stats.attestation.mean if stats.attestation else 0)
                training.append(stats.training.mean if stats.training else 0)
                inference.append(stats.inference.mean if stats.inference else 0)
        
        if not labels:
            return
            
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
        
        x = np.arange(len(labels))
        width = 0.6
        
        p1 = ax.bar(x, attestation, width, label='Attestation', color='#95a5a6')
        p2 = ax.bar(x, training, width, bottom=attestation, label='Training', color='#3498db')
        
        bottom2 = [a + t for a, t in zip(attestation, training)]
        p3 = ax.bar(x, inference, width, bottom=bottom2, label='Inference', color='#2ecc71')
        
        # Total time labels
        totals = [a + t + i for a, t, i in zip(attestation, training, inference)]
        for i, total in enumerate(totals):
            ax.annotate(f'{total:.0f} ms',
                       xy=(i, total),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center',
                       fontsize=FONT_SIZES['annotation'])
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Time (ms)')
        ax.set_title('Total Execution Time Breakdown')
        ax.legend(loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bar_total_time.pdf')
        plt.savefig(self.output_dir / 'bar_total_time.png')
        plt.close()
        
    def plot_combined_figure(self, report):
        """Combined figure for paper (single column)"""
        benchmarks = ["python", "cpp", "rust"]
        
        fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 7))
        
        # --- Subplot 1: Violin plot for training ---
        ax = axes[0]
        data, labels, colors = self._prepare_data(report, "training")
        
        if data:
            parts = ax.violinplot(data, positions=range(len(data)),
                                  showmeans=True, showmedians=True)
            
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
                
            for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
                if partname in parts:
                    parts[partname].set_edgecolor('#333333')
                    parts[partname].set_linewidth(1)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Time (ms)')
            ax.set_title('(a) Training Time Distribution')
            
            for i, d in enumerate(data):
                mean_val = np.mean(d)
                std_val = np.std(d)
                ax.annotate(f'μ={mean_val:.0f}\nσ={std_val:.0f}', 
                           xy=(i, max(d)),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=FONT_SIZES['annotation'])
        
        # --- Subplot 2: Violin plot for inference ---
        ax = axes[1]
        data, labels, colors = self._prepare_data(report, "inference")
        
        if data:
            parts = ax.violinplot(data, positions=range(len(data)),
                                  showmeans=True, showmedians=True)
            
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
                
            for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
                if partname in parts:
                    parts[partname].set_edgecolor('#333333')
                    parts[partname].set_linewidth(1)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Time (ms)')
            ax.set_title('(b) Inference Time Distribution')
            
            for i, d in enumerate(data):
                mean_val = np.mean(d)
                std_val = np.std(d)
                ax.annotate(f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                           xy=(i, max(d)),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=FONT_SIZES['annotation'])
        
        # --- Subplot 3: Stacked bar for total time ---
        ax = axes[2]
        
        labels_bar = []
        attestation = []
        training = []
        inference = []
        
        for name in benchmarks:
            if name in report.benchmarks:
                stats = report.benchmarks[name]
                labels_bar.append(DISPLAY_NAMES.get(name, name))
                attestation.append(stats.attestation.mean if stats.attestation else 0)
                training.append(stats.training.mean if stats.training else 0)
                inference.append(stats.inference.mean if stats.inference else 0)
        
        if labels_bar:
            x = np.arange(len(labels_bar))
            width = 0.6
            
            p1 = ax.bar(x, attestation, width, label='Attestation', color='#95a5a6')
            p2 = ax.bar(x, training, width, bottom=attestation, label='Training', color='#3498db')
            
            bottom2 = [a + t for a, t in zip(attestation, training)]
            p3 = ax.bar(x, inference, width, bottom=bottom2, label='Inference', color='#2ecc71')
            
            totals = [a + t + i for a, t, i in zip(attestation, training, inference)]
            for i, total in enumerate(totals):
                ax.annotate(f'{total:.0f} ms',
                           xy=(i, total),
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center',
                           fontsize=FONT_SIZES['annotation'])
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels_bar)
            ax.set_ylabel('Time (ms)')
            ax.set_title('(c) Total Execution Time Breakdown')
            ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_figure.pdf')
        plt.savefig(self.output_dir / 'combined_figure.png')
        plt.close()
        
        print(f"   📈 Combined figure: {self.output_dir / 'combined_figure.pdf'}")
