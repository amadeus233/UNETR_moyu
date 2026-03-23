"""
Reconstruct and visualize training metrics from conversation history
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MILESTONES = {
    1: 0.3736,
    2: 0.4726,
    24: 0.5633,
    48: 0.6022,
    73: 0.6344,
    93: 0.6376,
    100: 0.6376,
}

def reconstruct_curve():
    epochs = list(range(1, 101))
    dice_scores = []
    milestone_epochs = sorted(MILESTONES.keys())
    
    for epoch in epochs:
        if epoch in MILESTONES:
            dice_scores.append(MILESTONES[epoch])
        else:
            prev = max([m for m in milestone_epochs if m < epoch], default=milestone_epochs[0])
            next_m = min([m for m in milestone_epochs if m > epoch], default=milestone_epochs[-1])
            if prev == next_m:
                dice_scores.append(MILESTONES[prev])
            else:
                ratio = (epoch - prev) / (next_m - prev)
                interp = MILESTONES[prev] + ratio * (MILESTONES[next_m] - MILESTONES[prev])
                dice_scores.append(interp)
    
    return pd.DataFrame({'epoch': epochs, 'best_dice': dice_scores})

def create_plots(df):
    output_dir = "visualization_results/reconstructed_100_epochs"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UNETR-2D Training Performance - 100 Epochs (Reconstructed)', 
                 fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['best_dice'], 'b-o', linewidth=2, markersize=4)
    ax1.axhline(y=df['best_dice'].max(), color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score Progression')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, 100])
    
    for ep, dice in MILESTONES.items():
        marker = '*' if ep == 93 else 'o'
        size = 150 if ep == 93 else 80
        ax1.scatter([ep], [dice], c='red' if ep == 93 else 'orange', 
                   s=size, marker=marker, zorder=5)
        ax1.annotate(f'E{ep}:{dice:.3f}', xy=(ep, dice), xytext=(3, 3), 
                    textcoords='offset points', fontsize=7)
    
    ax2 = axes[0, 1]
    changes = df['best_dice'].diff()
    colors = ['green' if x > 0 else ('red' if x < 0 else 'gray') for x in changes]
    ax2.bar(df['epoch'], changes, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Change')
    ax2.set_title('Dice Score Change Rate')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    cumulative = df['best_dice'] - df['best_dice'].iloc[0]
    ax3.fill_between(df['epoch'], cumulative, alpha=0.4, color='purple')
    ax3.plot(df['epoch'], cumulative, 'purple', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Cumulative Improvement')
    ax3.set_title('Improvement from Baseline')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for w, c in [(5, 'blue'), (10, 'green'), (20, 'red')]:
        rolling = df['best_dice'].rolling(window=w, min_periods=1).mean()
        ax4.plot(df['epoch'], rolling, label=f'{w}-epoch MA', color=c, linewidth=2)
    ax4.plot(df['epoch'], df['best_dice'], 'gray', linewidth=1, alpha=0.5, linestyle='--')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Dice Score')
    ax4.set_title('Moving Average Trends')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    stats = (f'Start: {df["best_dice"].iloc[0]:.4f}\n'
             f'Final: {df["best_dice"].iloc[-1]:.4f}\n'
             f'Best: {df["best_dice"].max():.4f} (Ep {df["best_dice"].idxmax()+1})\n'
             f'Improvement: +{df["best_dice"].iloc[-1] - df["best_dice"].iloc[0]:.4f}')
    
    fig.text(0.02, 0.02, stats, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    return plot_path

def export_csv(df):
    output_dir = "visualization_results/reconstructed_100_epochs"
    os.makedirs(output_dir, exist_ok=True)
    
    df['change'] = df['best_dice'].diff()
    df['cumulative'] = df['best_dice'] - df['best_dice'].iloc[0]
    
    csv_path = os.path.join(output_dir, 'training_data.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved: {csv_path}")
    return csv_path

def create_report(df):
    output_dir = "visualization_results/reconstructed_100_epochs"
    os.makedirs(output_dir, exist_ok=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("UNETR-2D TRAINING ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now()}")
    lines.append("")
    lines.append("KEY METRICS:")
    lines.append(f"  Total Epochs:    100")
    lines.append(f"  Starting Dice:   {df['best_dice'].iloc[0]:.4f}")
    lines.append(f"  Final Dice:      {df['best_dice'].iloc[-1]:.4f}")
    lines.append(f"  Best Dice:       {df['best_dice'].max():.4f} (Epoch {df['best_dice'].idxmax()+1})")
    lines.append(f"  Improvement:     +{df['best_dice'].iloc[-1] - df['best_dice'].iloc[0]:.4f}")
    lines.append(f"  Increase:        {((df['best_dice'].iloc[-1]/df['best_dice'].iloc[0])-1)*100:.1f}%")
    lines.append("")
    lines.append("MILESTONES:")
    for ep, dice in sorted(MILESTONES.items()):
        lines.append(f"  Epoch {ep:3d}: {dice:.4f}")
    lines.append("")
    lines.append("=" * 80)
    
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {report_path}")
    return report_path

def main():
    print("Reconstructing training curve from history...")
    df = reconstruct_curve()
    
    print("Creating visualizations...")
    create_plots(df)
    
    print("Exporting data to CSV...")
    export_csv(df)
    
    print("Generating report...")
    create_report(df)
    
    print("\nDone! Check visualization_results/reconstructed_100_epochs/")

if __name__ == "__main__":
    main()
