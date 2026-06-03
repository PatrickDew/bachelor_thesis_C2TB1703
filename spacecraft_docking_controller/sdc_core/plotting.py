"""
Publication-quality plotting for spacecraft docking controller analysis.

Used by notebooks/docking_analysis.ipynb and scripts/plot_docking_data.py.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# IEEE / research-paper style defaults
PAPER_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

COLORS = {
    'range': '#1f4e79',
    'rate': '#c0392b',
    'pos_x': '#e67e22',
    'pos_y': '#27ae60',
    'pos_z': '#2980b9',
    'ctrl': '#8e44ad',
    'target': '#2c3e50',
    'vision': '#3498db',
    'gt': '#e74c3c',
}


def apply_paper_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(PAPER_STYLE)


def _save(fig, path: str, show: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {os.path.abspath(path)}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def _has_cols(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def plot_range_and_rate(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 1 — Range and closing velocity vs time."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)

    axes[0].plot(df['time'], df['range'], color=COLORS['range'], label='Range $\\rho$')
    axes[0].axhline(0, color=COLORS['target'], ls=':', lw=1, alpha=0.6)
    axes[0].set_ylabel('Range (m)')
    axes[0].set_title('(a) Relative range to docking port')
    axes[0].legend(loc='upper right')

    rate_cm = df['range_rate'] * 100
    axes[1].plot(df['time'], rate_cm, color=COLORS['rate'], label='Closing rate')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].fill_between(df['time'], rate_cm, 0, where=(rate_cm < 0),
                         color=COLORS['range'], alpha=0.15, label='Approaching')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Closing rate (cm/s)')
    axes[1].set_title('(b) Line-of-sight closing velocity')
    axes[1].legend(loc='upper right')

    fig.suptitle('Docking Approach — Range Profile', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig01_range_profile.png'), show)


def plot_position_errors(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 2 — Position components or tracking errors."""
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    labels = ['X (lateral)', 'Y (vertical)', 'Z (depth)']
    keys = ['x', 'y', 'z']
    colors = [COLORS['pos_x'], COLORS['pos_y'], COLORS['pos_z']]

    use_err = _has_cols(df, 'err_x', 'err_y', 'err_z')
    for ax, key, label, color in zip(axes, keys, labels, colors):
        col = f'err_{key}' if use_err else f'pos_{key}'
        ax.plot(df['time'], df[col], color=color, label=label)
        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_ylabel(f'{"Error" if use_err else "Pos."} {key.upper()} (m)')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    title = 'Tracking Error Components' if use_err else 'Relative Position (Camera Frame)'
    fig.suptitle(title, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig02_position_errors.png'), show)


def plot_trajectory_2d(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 3 — 2D approach trajectories (top + side view)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    xcol = 'err_x' if _has_cols(df, 'err_x') else 'pos_x'
    ycol = 'err_y' if _has_cols(df, 'err_y') else 'pos_y'
    zcol = 'err_z' if _has_cols(df, 'err_z') else 'pos_z'

    for ax, (xc, yc, title) in zip(
        axes,
        [(xcol, ycol, 'XY — Top View'), (zcol, xcol, 'XZ — Side View (Depth vs Lateral')],
    ):
        sc = ax.scatter(df[xc], df[yc], c=df['time'], cmap='viridis', s=8, alpha=0.75)
        ax.plot(df[xc].iloc[0], df[yc].iloc[0], 'o', color=COLORS['pos_y'],
                ms=8, label='Start', zorder=5)
        ax.plot(df[xc].iloc[-1], df[yc].iloc[-1], '*', color=COLORS['rate'],
                ms=12, label='End', zorder=5)
        ax.plot(0, 0, '+', color=COLORS['target'], ms=14, mew=2, label='Target', zorder=5)
        ax.set_xlabel(f'{xc.replace("_", " ").title()} (m)')
        ax.set_ylabel(f'{yc.replace("_", " ").title()} (m)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        plt.colorbar(sc, ax=ax, label='Time (s)', shrink=0.85)

    fig.suptitle('Approach Trajectory — Camera Frame', fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig03_trajectory_2d.png'), show)


def plot_control_effort(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 4 — PID acceleration commands and Isaac forces."""
    has_force = _has_cols(df, 'force_isaac_x')
    nrows = 2 if has_force else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(7.5, 3.5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    axes[0].plot(df['time'], df['ctrl_x'] * 1000, label='$a_x$', color=COLORS['pos_x'])
    axes[0].plot(df['time'], df['ctrl_y'] * 1000, label='$a_y$', color=COLORS['pos_y'])
    axes[0].plot(df['time'], df['ctrl_z'] * 1000, label='$a_z$', color=COLORS['pos_z'])
    axes[0].axhline(0, color='black', lw=0.6)
    axes[0].set_ylabel('Accel. (mm/s²)')
    axes[0].set_title('(a) PID command — camera frame')
    axes[0].legend(loc='upper right', ncol=3)

    if has_force:
        axes[1].plot(df['time'], df['force_isaac_x'], label='$F_x$', color=COLORS['pos_x'])
        axes[1].plot(df['time'], df['force_isaac_y'], label='$F_y$', color=COLORS['pos_y'])
        axes[1].plot(df['time'], df['force_isaac_z'], label='$F_z$', color=COLORS['pos_z'])
        axes[1].axhline(0, color='black', lw=0.6)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Force (N)')
        axes[1].set_title('(b) Applied force — Isaac frame')
        axes[1].legend(loc='upper right', ncol=3)
    else:
        axes[0].set_xlabel('Time (s)')

    fig.suptitle('Control Effort', fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig04_control_effort.png'), show)


def plot_phase_plane(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 5 — Phase plane: range vs closing rate (standard AR&D plot)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        df['range'], df['range_rate'] * 100,
        c=df['time'], cmap='plasma', s=12, alpha=0.8,
    )
    ax.plot(df['range'].iloc[0], df['range_rate'].iloc[0] * 100,
            'o', color=COLORS['pos_y'], ms=8, label='Start')
    ax.plot(df['range'].iloc[-1], df['range_rate'].iloc[-1] * 100,
            '*', color=COLORS['rate'], ms=12, label='End')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(0, color=COLORS['target'], ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Range $\\rho$ (m)')
    ax.set_ylabel('Closing rate $\\dot{\\rho}$ (cm/s)')
    ax.set_title('Phase Plane — Range vs Closing Rate')
    ax.legend(loc='best')
    plt.colorbar(sc, label='Time (s)')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig05_phase_plane.png'), show)


def plot_pid_decomposition(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 6 — PID term decomposition (depth axis)."""
    if not _has_cols(df, 'pid_p_z', 'pid_i_z', 'pid_d_z'):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(df['time'], df['pid_p_z'] * 1000, label='P (depth)', color=COLORS['pos_z'])
    ax.plot(df['time'], df['pid_i_z'] * 1000, label='I (depth)', color=COLORS['ctrl'])
    ax.plot(df['time'], df['pid_d_z'] * 1000, label='D (depth)', color=COLORS['rate'])
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Contribution (mm/s²)')
    ax.set_title('PID Term Decomposition — Depth Axis (Z)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig06_pid_decomposition.png'), show)


def plot_vision_vs_gt(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Fig. 7 — Vision pose vs Isaac ground truth (if available)."""
    if not _has_cols(df, 'gt_pos_x') or df['gt_pos_x'].notna().sum() < 5:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True)
    for ax, key, label in zip(axes, ['x', 'y', 'z'], ['X', 'Y', 'Z']):
        ax.plot(df['time'], df[f'pos_{key}'], color=COLORS['vision'],
                label='Vision (camera frame)', lw=1.8)
        ax.plot(df['time'], df[f'gt_pos_{key}'], color=COLORS['gt'],
                label='Isaac GT', lw=1.8, ls='--')
        ax.set_ylabel(f'{label} (m)')
        ax.legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Vision Pose vs Isaac Sim Ground Truth', fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig07_vision_vs_gt.png'), show)


def plot_paper_dashboard(df: pd.DataFrame, output_dir: str, show: bool = False) -> str:
    """Combined multi-panel figure suitable for thesis / paper."""
    apply_paper_style()
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.32)

    # Range
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['time'], df['range'], color=COLORS['range'], lw=2)
    ax1.fill_between(df['time'], df['range'], alpha=0.12, color=COLORS['range'])
    ax1.set_ylabel('Range (m)')
    ax1.set_title('(a) Approach range')

    # Closing rate
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df['time'], df['range_rate'] * 100, color=COLORS['rate'], lw=2)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_ylabel('Rate (cm/s)')
    ax2.set_title('(b) Closing rate')

    # Position errors
    ax3 = fig.add_subplot(gs[1, 0])
    xcol = 'err_x' if _has_cols(df, 'err_x') else 'pos_x'
    ycol = 'err_y' if _has_cols(df, 'err_y') else 'pos_y'
    zcol = 'err_z' if _has_cols(df, 'err_z') else 'pos_z'
    ax3.plot(df['time'], df[xcol], label='X', color=COLORS['pos_x'])
    ax3.plot(df['time'], df[ycol], label='Y', color=COLORS['pos_y'])
    ax3.plot(df['time'], df[zcol], label='Z', color=COLORS['pos_z'])
    ax3.axhline(0, color='gray', ls='--', lw=0.7)
    ax3.set_ylabel('Error (m)')
    ax3.set_title('(c) Position error')
    ax3.legend(fontsize=8)

    # Control
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['time'], df['ctrl_z'] * 1000, color=COLORS['pos_z'], label='$a_z$')
    ax4.axhline(0, color='k', lw=0.6)
    ax4.set_ylabel('mm/s²')
    ax4.set_title('(d) Depth control')
    ax4.legend(fontsize=8)

    # Phase plane
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(df['range'], df['range_rate'] * 100,
                c=df['time'], cmap='plasma', s=6, alpha=0.7)
    ax5.set_xlabel('Range (m)')
    ax5.set_ylabel('Rate (cm/s)')
    ax5.set_title('(e) Phase plane')

    # 2D trajectory
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(df[zcol], df[xcol], c=df['time'], cmap='viridis', s=6, alpha=0.7)
    ax6.plot(0, 0, '+', color=COLORS['target'], ms=12, mew=2)
    ax6.set_xlabel('Z depth (m)')
    ax6.set_ylabel('X lateral (m)')
    ax6.set_title('(f) Approach path')

    # 3D trajectory (fallback to 2D if 3D projection unavailable)
    ax7 = fig.add_subplot(gs[2, 1:])
    try:
        ax7.remove()
        ax7 = fig.add_subplot(gs[2, 1:], projection='3d')
        ax7.plot(df[xcol], df[ycol], df[zcol], color=COLORS['range'], lw=1.2, alpha=0.8)
        ax7.scatter([0], [0], [0], color=COLORS['target'], s=60, marker='+', linewidths=2)
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_zlabel('Z')
        ax7.set_title('(g) 3D trajectory')
    except Exception:
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.plot(df[zcol], df[xcol], color=COLORS['range'], lw=1.5)
        ax7.plot(0, 0, '+', color=COLORS['target'], ms=12, mew=2)
        ax7.set_xlabel('Z depth (m)')
        ax7.set_ylabel('X lateral (m)')
        ax7.set_title('(g) Side view (3D unavailable)')

    # Summary stats box
    initial = df['range'].iloc[0]
    final = df['range'].iloc[-1]
    duration = df['time'].max()
    stats = (
        f'Initial range: {initial:.3f} m\n'
        f'Final range:   {final:.3f} m\n'
        f'Duration:      {duration:.1f} s\n'
        f'Max |rate|:    {df["range_rate"].abs().max()*100:.2f} cm/s'
    )
    fig.text(0.02, 0.02, stats, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))

    fig.suptitle(
        'Closed-Loop PID Docking — Isaac Sim + Vision Pose',
        fontsize=14, fontweight='bold', y=0.98,
    )

    out_png = os.path.join(output_dir, 'docking_paper_figure.png')
    out_pdf = os.path.join(output_dir, 'docking_paper_figure.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved: {os.path.abspath(out_png)}')
    print(f'Saved: {os.path.abspath(out_pdf)}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_png


def generate_all_figures(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Generate full figure set for a docking run."""
    apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    plot_range_and_rate(df, output_dir, show)
    plot_position_errors(df, output_dir, show)
    plot_trajectory_2d(df, output_dir, show)
    plot_control_effort(df, output_dir, show)
    plot_phase_plane(df, output_dir, show)
    plot_pid_decomposition(df, output_dir, show)
    plot_vision_vs_gt(df, output_dir, show)
    plot_paper_dashboard(df, output_dir, show)


def write_summary(df: pd.DataFrame, output_dir: str) -> str:
    """Write text summary for thesis results section."""
    initial = float(df['range'].iloc[0])
    final = float(df['range'].iloc[-1])
    duration = float(df['time'].max())
    lines = [
        'Docking Run Summary (PID Closed-Loop)',
        '=' * 42,
        f'Initial range:     {initial:.4f} m',
        f'Final range:       {final:.4f} m',
        f'Range reduction:   {(1 - final/initial)*100:.1f}%' if initial > 0 else '',
        f'Duration:          {duration:.1f} s',
        f'Mean closing rate: {df["range_rate"].mean()*100:.3f} cm/s',
        f'Max |closing|:    {df["range_rate"].abs().max()*100:.3f} cm/s',
    ]
    if _has_cols(df, 'err_x'):
        lines += [
            '',
            'Final tracking error (camera frame):',
            f'  e_x = {df["err_x"].iloc[-1]:.4f} m',
            f'  e_y = {df["err_y"].iloc[-1]:.4f} m',
            f'  e_z = {df["err_z"].iloc[-1]:.4f} m',
        ]
    text = '\n'.join(lines)
    path = os.path.join(output_dir, 'docking_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    return path
