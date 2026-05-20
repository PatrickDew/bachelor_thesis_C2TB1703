#!/usr/bin/env python3
"""
Plot docking controller data from CSV log file.

Usage:
    python3 plot_docking_data.py [log_file_path]
    
    Default log file: /tmp/docking_data.csv

Generates publication-quality plots for research presentations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (14, 10),
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def load_data(filepath):
    """Load CSV data from log file."""
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return data


def plot_docking_performance(data, output_dir=None):
    """Generate comprehensive docking performance plots."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Colors for consistent styling
    colors = {
        'range': '#2E86AB',
        'rate': '#A23B72',
        'pos_x': '#F18F01',
        'pos_y': '#C73E1D',
        'pos_z': '#3B1F2B',
        'ctrl': '#44803F',
        'twist': '#6B4E71'
    }
    
    # 1. Range vs Time (main plot)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data['time'], data['range'], color=colors['range'], linewidth=2.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Range to Target (m)')
    ax1.set_title('Docking Approach - Range vs Time', fontweight='bold')
    ax1.fill_between(data['time'], data['range'], alpha=0.3, color=colors['range'])
    
    # Add final range annotation
    final_range = data['range'][-1]
    ax1.axhline(y=final_range, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(f'Final: {final_range:.3f}m', 
                xy=(data['time'][-1], final_range),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=11, color=colors['range'])
    
    # 2. Range Rate vs Time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(data['time'], data['range_rate'] * 100, color=colors['rate'], linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range Rate (cm/s)')
    ax2.set_title('Approach Velocity')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. 3D Position
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(data['time'], data['pos_x'], label='X', color=colors['pos_x'])
    ax3.plot(data['time'], data['pos_y'], label='Y', color=colors['pos_y'])
    ax3.plot(data['time'], data['pos_z'], label='Z', color=colors['pos_z'])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position Components')
    ax3.legend(loc='upper right')
    
    # 4. Control Commands
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(data['time'], data['ctrl_x'] * 1000, label='Ctrl X', alpha=0.8)
    ax4.plot(data['time'], data['ctrl_y'] * 1000, label='Ctrl Y', alpha=0.8)
    ax4.plot(data['time'], data['ctrl_z'] * 1000, label='Ctrl Z', alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control (mm/s²)')
    ax4.set_title('PID Control Output')
    ax4.legend(loc='upper right')
    
    # 5. Twist Commands (velocity sent to Isaac Sim)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(data['time'], data['twist_x'] * 100, label='Twist X', alpha=0.8)
    ax5.plot(data['time'], data['twist_y'] * 100, label='Twist Y', alpha=0.8)
    ax5.plot(data['time'], data['twist_z'] * 100, label='Twist Z', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity Cmd (cm/s)')
    ax5.set_title('Velocity Commands to Isaac Sim')
    ax5.legend(loc='upper right')
    
    # 6. XY Position Trajectory
    ax6 = fig.add_subplot(gs[2, 0])
    scatter = ax6.scatter(data['pos_x'], data['pos_y'], 
                         c=data['time'], cmap='viridis', s=10, alpha=0.7)
    ax6.plot(data['pos_x'][0], data['pos_y'][0], 'go', markersize=12, label='Start')
    ax6.plot(data['pos_x'][-1], data['pos_y'][-1], 'r*', markersize=15, label='End')
    ax6.plot(0, 0, 'k+', markersize=15, markeredgewidth=3, label='Target')
    ax6.set_xlabel('X Position (m)')
    ax6.set_ylabel('Y Position (m)')
    ax6.set_title('XY Trajectory (Top View)')
    ax6.legend(loc='upper right')
    ax6.axis('equal')
    plt.colorbar(scatter, ax=ax6, label='Time (s)')
    
    # 7. XZ Position Trajectory (side view - approach)
    ax7 = fig.add_subplot(gs[2, 1])
    scatter2 = ax7.scatter(data['pos_z'], data['pos_x'], 
                          c=data['time'], cmap='viridis', s=10, alpha=0.7)
    ax7.plot(data['pos_z'][0], data['pos_x'][0], 'go', markersize=12, label='Start')
    ax7.plot(data['pos_z'][-1], data['pos_x'][-1], 'r*', markersize=15, label='End')
    ax7.plot(0, 0, 'k+', markersize=15, markeredgewidth=3, label='Target')
    ax7.set_xlabel('Z Position - Depth (m)')
    ax7.set_ylabel('X Position (m)')
    ax7.set_title('XZ Trajectory (Side View)')
    ax7.legend(loc='upper right')
    plt.colorbar(scatter2, ax=ax7, label='Time (s)')
    
    # 8. Summary Statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate statistics
    initial_range = data['range'][0]
    final_range = data['range'][-1]
    range_reduction = (1 - final_range/initial_range) * 100
    total_time = data['time'][-1]
    avg_rate = (initial_range - final_range) / total_time * 100  # cm/s
    
    stats_text = f"""
    Docking Performance Summary
    ══════════════════════════════
    
    Initial Range:     {initial_range:.3f} m
    Final Range:       {final_range:.3f} m
    Range Reduction:   {range_reduction:.1f}%
    
    Total Time:        {total_time:.1f} s
    Avg Approach Rate: {avg_rate:.2f} cm/s
    
    Max |Control|:     {np.max(np.abs(data['ctrl_x'])):.4f} m/s²
    Max |Twist|:       {np.max(np.abs(data['twist_x'])):.4f} m/s
    """
    
    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Spacecraft Docking Controller - PID Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure
    if output_dir:
        output_path = os.path.join(output_dir, 'docking_performance.png')
    else:
        output_path = '/tmp/docking_performance.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f'Plot saved to: {output_path}')
    
    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'PDF saved to: {pdf_path}')
    
    plt.show()


def main():
    # Get log file path from argument or use default
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = '/tmp/docking_data.csv'
    
    if not os.path.exists(log_file):
        print(f'Error: Log file not found: {log_file}')
        print('Make sure to enable data logging in docking_params.yaml:')
        print('  enable_data_logging: true')
        print('  log_file_path: "/tmp/docking_data.csv"')
        sys.exit(1)
    
    print(f'Loading data from: {log_file}')
    data = load_data(log_file)
    print(f'Loaded {len(data)} data points')
    
    plot_docking_performance(data)


if __name__ == '__main__':
    main()


