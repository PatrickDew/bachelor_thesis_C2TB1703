#!/usr/bin/env python3
"""
Plot docking controller CSV logs (same figures as notebooks/docking_analysis.ipynb).

Usage:
    python plot_docking_data.py --latest --all
    python plot_docking_data.py logs/docking_YYYYMMDD_HHMMSS.csv
    python plot_docking_data.py --latest --compare

Outputs in the log folder (or --output-dir):
    plot_range_vs_time.png
    plot_range_rate.png
    plot_position_components.png
    plot_trajectory_2d.png
    plot_control_outputs.png
    plot_control_theory_pose_vs_command.png
    plot_control_theory_command_vs_error.png
    plot_control_theory_actuation_vs_response.png
    (and other plot_control_theory_*.png — see notebook §8)
    plot_pid_control.png
    plot_twist_commands.png
    plot_trajectory_3d.png
    plot_comparison.png          (with --compare, if 2+ logs)
    docking_performance.png/pdf  (combined dashboard, unless --no-combined)
    docking_summary.txt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from sdc_core.log_paths import find_latest_log, log_search_directories
except ImportError:
    import glob as _glob

    def _resolve_logs_dir(caller_file=None):
        env = os.environ.get('DOCKING_LOG_DIRECTORY', '').strip()
        if env:
            return os.path.abspath(os.path.expanduser(env))
        script_dir = os.path.dirname(os.path.realpath(caller_file or __file__))
        return os.path.abspath(os.path.join(script_dir, '..', 'logs'))

    def log_search_directories(configured='', caller_file=None):
        return [_resolve_logs_dir(caller_file)]

    def find_latest_log(configured='', caller_file=None):
        logs_dir = _resolve_logs_dir(caller_file)
        files = _glob.glob(os.path.join(logs_dir, 'docking_*.csv'))
        return max(files, key=os.path.getmtime) if files else None

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (14, 10),
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _save(fig, path: str, show: bool) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved: {os.path.abspath(path)}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary(df: pd.DataFrame, output_dir: str) -> str:
    initial_range = float(df['range'].iloc[0])
    final_range = float(df['range'].iloc[-1])
    min_range = float(df['range'].min())
    total_time = float(df['time'].max())
    avg_rate = (initial_range - final_range) / total_time if total_time > 0 else 0.0

    lines = [
        'Docking run summary',
        '===================',
        '',
        'RANGE',
        f'  Initial:   {initial_range:.4f} m',
        f'  Final:     {final_range:.4f} m',
        f'  Minimum:   {min_range:.4f} m',
        f'  Reduction: {(1 - final_range / initial_range) * 100:.1f}%',
        '',
        'TIME',
        f'  Duration:    {total_time:.1f} s',
        f'  Data points: {len(df)}',
        '',
        'VELOCITY',
        f'  Avg approach rate: {avg_rate * 100:.3f} cm/s',
        f'  Max |range_rate|:  {df["range_rate"].abs().max() * 100:.3f} cm/s',
        '',
        'CONTROL',
        f'  Max |ctrl_x|: {df["ctrl_x"].abs().max() * 1000:.3f} mm/s^2',
        f'  Max |ctrl_y|: {df["ctrl_y"].abs().max() * 1000:.3f} mm/s^2',
        f'  Max |ctrl_z|: {df["ctrl_z"].abs().max() * 1000:.3f} mm/s^2',
        '',
        'POSITION (final)',
        f'  X: {df["pos_x"].iloc[-1]:.4f} m',
        f'  Y: {df["pos_y"].iloc[-1]:.4f} m',
        f'  Z: {df["pos_z"].iloc[-1]:.4f} m',
    ]
    text = '\n'.join(lines)
    path = os.path.join(output_dir, 'docking_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Saved: {os.path.abspath(path)}')
    print('\n' + text)
    return path


def plot_range_vs_time(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['time'], df['range'], color='#2E86AB', linewidth=2.5, label='Range')
    ax.fill_between(df['time'], df['range'], alpha=0.3, color='#2E86AB')
    final_range = df['range'].iloc[-1]
    ax.axhline(y=final_range, color='gray', linestyle='--', alpha=0.5,
               label=f'Final: {final_range:.3f}m')
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.3, label='Target (0m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range to Target')
    ax.set_title('Spacecraft Docking Approach - Range vs Time', fontweight='bold', fontsize=16)
    ax.legend(loc='upper right')
    ax.set_xlim(0, df['time'].max())
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_range_vs_time.png'), show)


def plot_range_rate(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    rate_cms = df['range_rate'] * 100
    ax.plot(df['time'], rate_cms, color='#A23B72', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.fill_between(df['time'], rate_cms, 0, where=(rate_cms < 0),
                    color='green', alpha=0.3, label='Approaching')
    ax.fill_between(df['time'], rate_cms, 0, where=(rate_cms > 0),
                    color='red', alpha=0.3, label='Moving away')
    avg_rate = rate_cms.mean()
    ax.axhline(y=avg_rate, color='blue', linestyle='--', alpha=0.5,
               label=f'Avg: {avg_rate:.2f} cm/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range Rate')
    ax.set_title('Approach Velocity', fontweight='bold')
    ax.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_range_rate.png'), show)


def plot_position_components(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {'x': '#F18F01', 'y': '#C73E1D', 'z': '#2E86AB'}
    for ax, (axis, color) in zip(axes, colors.items()):
        col = f'pos_{axis}'
        ax.plot(df['time'], df[col], color=color, linewidth=2, label=f'{axis.upper()} position')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel(axis.upper())
        ax.legend(loc='upper right')
        ax.fill_between(df['time'], df[col], 0, alpha=0.2, color=color)
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Position Components Over Time', fontweight='bold', fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_position_components.png'), show)


def plot_trajectory_2d(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    sc1 = ax1.scatter(df['pos_x'], df['pos_y'], c=df['time'], cmap='viridis', s=15, alpha=0.7)
    ax1.plot(df['pos_x'].iloc[0], df['pos_y'].iloc[0], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], 'r*', markersize=15, label='End', zorder=5)
    ax1.plot(0, 0, 'k+', markersize=20, markeredgewidth=3, label='Target', zorder=5)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('XY Trajectory (Top View)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.axis('equal')
    plt.colorbar(sc1, ax=ax1, label='Time (s)')

    ax2 = axes[1]
    sc2 = ax2.scatter(df['pos_z'], df['pos_x'], c=df['time'], cmap='viridis', s=15, alpha=0.7)
    ax2.plot(df['pos_z'].iloc[0], df['pos_x'].iloc[0], 'go', markersize=12, label='Start', zorder=5)
    ax2.plot(df['pos_z'].iloc[-1], df['pos_x'].iloc[-1], 'r*', markersize=15, label='End', zorder=5)
    ax2.plot(0, 0, 'k+', markersize=20, markeredgewidth=3, label='Target', zorder=5)
    ax2.set_xlabel('Z Position - Depth (m)')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('XZ Trajectory (Side View - Approach)', fontweight='bold')
    ax2.legend(loc='upper right')
    plt.colorbar(sc2, ax=ax2, label='Time (s)')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_trajectory_2d.png'), show)


def plot_control_outputs(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1 = axes[0]
    ax1.plot(df['time'], df['ctrl_x'] * 1000, label='Ctrl X', alpha=0.8)
    ax1.plot(df['time'], df['ctrl_y'] * 1000, label='Ctrl Y', alpha=0.8)
    ax1.plot(df['time'], df['ctrl_z'] * 1000, label='Ctrl Z', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('PID Control')
    ax1.set_title('PID Controller Output', fontweight='bold')
    ax1.legend(loc='lower right')

    ax2 = axes[1]
    ax2.plot(df['time'], df['twist_x'] * 100, label='Twist X (fwd)', alpha=0.8)
    ax2.plot(df['time'], df['twist_y'] * 100, label='Twist Y (left)', alpha=0.8)
    ax2.plot(df['time'], df['twist_z'] * 100, label='Twist Z (up)', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity Command')
    ax2.set_title('Velocity Commands to Isaac Sim', fontweight='bold')
    ax2.legend(loc='lower right')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_control_outputs.png'), show)


def plot_pid_control(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['time'], df['ctrl_x'] * 1000, label='Ctrl X', alpha=0.8, color='#F18F01')
    ax.plot(df['time'], df['ctrl_y'] * 1000, label='Ctrl Y', alpha=0.8, color='#C73E1D')
    ax.plot(df['time'], df['ctrl_z'] * 1000, label='Ctrl Z', alpha=0.8, color='#2E86AB')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PID Control Output (mm/s^2)')
    ax.set_title('PID Controller Output', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_pid_control.png'), show)


def plot_twist_commands(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['time'], df['twist_x'] * 100, label='Twist X (fwd)', alpha=0.8, color='#F18F01')
    ax.plot(df['time'], df['twist_y'] * 100, label='Twist Y (left)', alpha=0.8, color='#C73E1D')
    ax.plot(df['time'], df['twist_z'] * 100, label='Twist Z (up)', alpha=0.8, color='#2E86AB')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Command (cm/s)')
    ax.set_title('Velocity Commands to Isaac Sim', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_twist_commands.png'), show)


def plot_trajectory_3d(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df['pos_x'], df['pos_y'], df['pos_z'],
        c=df['time'], cmap='plasma', s=20, alpha=0.7,
    )
    ax.scatter(
        df['pos_x'].iloc[0], df['pos_y'].iloc[0], df['pos_z'].iloc[0],
        c='green', s=200, marker='o', label='Start', edgecolor='black',
    )
    ax.scatter(
        df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], df['pos_z'].iloc[-1],
        c='red', s=300, marker='*', label='End', edgecolor='black',
    )
    ax.scatter(0, 0, 0, c='black', s=300, marker='+', linewidths=3, label='Target (0,0,0)')
    ax.plot(
        [df['pos_x'].iloc[-1], 0], [df['pos_y'].iloc[-1], 0], [df['pos_z'].iloc[-1], 0],
        'k--', alpha=0.3, linewidth=2,
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Docking Trajectory', fontweight='bold', fontsize=16)
    ax.legend(loc='upper left')
    plt.colorbar(scatter, label='Time (s)', shrink=0.6, pad=0.1)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_trajectory_3d.png'), show)


def plot_comparison(log_paths: list[str], output_dir: str, show: bool) -> None:
    if len(log_paths) < 2:
        print('Need at least 2 docking_*.csv files for comparison plot.')
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    for i, path in enumerate(log_paths):
        data = load_data(path)
        short_name = os.path.basename(path).replace('docking_', '').replace('.csv', '')
        ax.plot(
            data['time'], data['range'],
            color=colors[i % len(colors)], linewidth=2,
            label=short_name, alpha=0.8,
        )
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.3, label='Target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Range Comparison Across Runs', fontweight='bold', fontsize=16)
    ax.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'plot_comparison.png'), show)


def plot_all_individual(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    """All per-plot PNGs from docking_analysis.ipynb."""
    plot_range_vs_time(df, output_dir, show)
    plot_range_rate(df, output_dir, show)
    plot_position_components(df, output_dir, show)
    plot_trajectory_2d(df, output_dir, show)
    plot_control_outputs(df, output_dir, show)
    plot_pid_control(df, output_dir, show)
    plot_twist_commands(df, output_dir, show)
    plot_trajectory_3d(df, output_dir, show)


def plot_docking_performance(df: pd.DataFrame, output_dir: str = '.', show: bool = True) -> str:
    """Combined multi-panel dashboard."""
    from matplotlib.gridspec import GridSpec

    data = df
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    colors = {
        'range': '#2E86AB', 'rate': '#A23B72', 'pos_x': '#F18F01',
        'pos_y': '#C73E1D', 'pos_z': '#3B1F2B',
    }

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data['time'], data['range'], color=colors['range'], linewidth=2.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Range to Target (m)')
    ax1.set_title('Docking Approach - Range vs Time', fontweight='bold')
    ax1.fill_between(data['time'], data['range'], alpha=0.3, color=colors['range'])
    final_range = data['range'].iloc[-1]
    ax1.axhline(y=final_range, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(
        f'Final: {final_range:.3f}m',
        xy=(data['time'].iloc[-1], final_range),
        xytext=(-60, 10), textcoords='offset points',
        fontsize=11, color=colors['range'],
    )

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(data['time'], data['range_rate'] * 100, color=colors['rate'], linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range Rate (cm/s)')
    ax2.set_title('Approach Velocity')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(data['time'], data['pos_x'], label='X', color=colors['pos_x'])
    ax3.plot(data['time'], data['pos_y'], label='Y', color=colors['pos_y'])
    ax3.plot(data['time'], data['pos_z'], label='Z', color=colors['pos_z'])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position Components')
    ax3.legend(loc='upper right')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(data['time'], data['ctrl_x'] * 1000, label='Ctrl X', alpha=0.8)
    ax4.plot(data['time'], data['ctrl_y'] * 1000, label='Ctrl Y', alpha=0.8)
    ax4.plot(data['time'], data['ctrl_z'] * 1000, label='Ctrl Z', alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control (mm/s^2)')
    ax4.set_title('PID Control Output')
    ax4.legend(loc='upper right')

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(data['time'], data['twist_x'] * 100, label='Twist X', alpha=0.8)
    ax5.plot(data['time'], data['twist_y'] * 100, label='Twist Y', alpha=0.8)
    ax5.plot(data['time'], data['twist_z'] * 100, label='Twist Z', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity Cmd (cm/s)')
    ax5.set_title('Velocity Commands to Isaac Sim')
    ax5.legend(loc='upper right')

    ax6 = fig.add_subplot(gs[2, 0])
    scatter = ax6.scatter(data['pos_x'], data['pos_y'], c=data['time'], cmap='viridis', s=10, alpha=0.7)
    ax6.plot(data['pos_x'].iloc[0], data['pos_y'].iloc[0], 'go', markersize=12, label='Start')
    ax6.plot(data['pos_x'].iloc[-1], data['pos_y'].iloc[-1], 'r*', markersize=15, label='End')
    ax6.plot(0, 0, 'k+', markersize=15, markeredgewidth=3, label='Target')
    ax6.set_xlabel('X Position (m)')
    ax6.set_ylabel('Y Position (m)')
    ax6.set_title('XY Trajectory (Top View)')
    ax6.legend(loc='upper right')
    ax6.axis('equal')
    plt.colorbar(scatter, ax=ax6, label='Time (s)')

    ax7 = fig.add_subplot(gs[2, 1])
    scatter2 = ax7.scatter(data['pos_z'], data['pos_x'], c=data['time'], cmap='viridis', s=10, alpha=0.7)
    ax7.plot(data['pos_z'].iloc[0], data['pos_x'].iloc[0], 'go', markersize=12, label='Start')
    ax7.plot(data['pos_z'].iloc[-1], data['pos_x'].iloc[-1], 'r*', markersize=15, label='End')
    ax7.plot(0, 0, 'k+', markersize=15, markeredgewidth=3, label='Target')
    ax7.set_xlabel('Z Position - Depth (m)')
    ax7.set_ylabel('X Position (m)')
    ax7.set_title('XZ Trajectory (Side View)')
    ax7.legend(loc='upper right')
    plt.colorbar(scatter2, ax=ax7, label='Time (s)')

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    initial_range = data['range'].iloc[0]
    final_range = data['range'].iloc[-1]
    range_reduction = (1 - final_range / initial_range) * 100
    total_time = data['time'].iloc[-1]
    avg_rate = (initial_range - final_range) / total_time * 100 if total_time > 0 else 0.0
    stats_text = f"""
    Docking Performance Summary
    ==============================

    Initial Range:     {initial_range:.3f} m
    Final Range:       {final_range:.3f} m
    Range Reduction:   {range_reduction:.1f}%

    Total Time:        {total_time:.1f} s
    Avg Approach Rate: {avg_rate:.2f} cm/s

    Max |Control|:     {data['ctrl_x'].abs().max():.4f} m/s^2
    Max |Twist|:       {data['twist_x'].abs().max():.4f} m/s
    """
    ax8.text(
        0.1, 0.9, stats_text, transform=ax8.transAxes,
        fontsize=12, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
    )

    plt.suptitle(
        'Spacecraft Docking Controller - Performance Analysis',
        fontsize=18, fontweight='bold', y=0.98,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'docking_performance.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = os.path.join(output_dir, 'docking_performance.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved: {os.path.abspath(output_path)}')
    print(f'Saved: {os.path.abspath(pdf_path)}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot docking logs (notebook-style individual PNGs + optional dashboard)',
    )
    parser.add_argument('log_file', nargs='?', help='Path to docking_YYYYMMDD_HHMMSS.csv')
    parser.add_argument('--latest', action='store_true', help='Use newest docking_*.csv')
    parser.add_argument('-o', '--output-dir', default=None, help='Output folder (default: log folder)')
    parser.add_argument('--no-show', action='store_true', help='Do not open plot windows')
    parser.add_argument(
        '--combined-only', action='store_true',
        help='Only docking_performance.png (skip individual notebook plots)',
    )
    parser.add_argument(
        '--no-combined', action='store_true',
        help='Skip combined docking_performance.png/pdf',
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Also plot range comparison for all docking_*.csv in the log folder',
    )
    args = parser.parse_args()

    show = not args.no_show

    if args.latest or not args.log_file:
        log_file = find_latest_log(caller_file=__file__)
        if not log_file:
            searched = ', '.join(log_search_directories(caller_file=__file__))
            print(f'No docking_*.csv found. Searched: {searched}')
            sys.exit(1)
        print(f'Using latest log: {log_file}')
    else:
        log_file = os.path.abspath(args.log_file)

    if not os.path.isfile(log_file):
        print(f'Error: Log file not found: {log_file}')
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(log_file) or '.'
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading data from: {log_file}')
    df = load_data(log_file)
    print(f'Loaded {len(df)} data points')

    write_summary(df, output_dir)

    if not args.combined_only:
        print('\n--- Individual analysis plots (notebook style) ---')
        plot_all_individual(df, output_dir, show=False)

    if not args.no_combined:
        print('\n--- Combined dashboard ---')
        plot_docking_performance(df, output_dir=output_dir, show=show)

    if args.compare:
        log_dir = os.path.dirname(log_file)
        all_logs = sorted(glob.glob(os.path.join(log_dir, 'docking_*.csv')))
        print('\n--- Multi-run comparison ---')
        plot_comparison(all_logs, output_dir, show=False)

    print(f'\nDone. All outputs in: {os.path.abspath(output_dir)}')


if __name__ == '__main__':
    main()
