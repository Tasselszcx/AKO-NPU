#!/usr/bin/env python3
"""
绘制 AKO-NPU 优化轨迹图（AKO4ALL 风格）
"""

import csv
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                it = int(r['iter'])
                rt = float(r['runtime_us']) if r.get('runtime_us') and r['runtime_us'].strip() not in ('', '-') else None
                sp = float(r['speedup']) if r.get('speedup') and r['speedup'].strip() not in ('', '-') else None
                st = r.get('status', '').strip()
                lb = r.get('label', '').strip()
                rows.append({'iter': it, 'runtime_us': rt, 'speedup': sp, 'status': st, 'label': lb})
            except (ValueError, KeyError):
                continue
    return rows


def auto_phases(rows):
    """把迭代均分成 4-6 个阶段"""
    if not rows:
        return []
    min_it = rows[0]['iter']
    max_it = rows[-1]['iter']
    span = max_it - min_it
    if span <= 0:
        return [(min_it, max_it, '')]

    # 找 frontier 跳跃点作为阶段分界
    frontier = 0
    jumps = []
    for r in rows:
        sp = r['speedup']
        if sp is not None and sp > frontier * 1.15 and frontier > 0:
            jumps.append(r['iter'])
        if sp is not None and sp > frontier:
            frontier = sp

    # 用跳跃点分阶段，如果太少就均分
    if len(jumps) < 2:
        n = min(4, max(2, len(rows) // 10))
        step = span / n
        boundaries = [int(min_it + step * i) for i in range(1, n)]
    else:
        # 选最大的几个跳跃
        boundaries = sorted(jumps[:5])

    # 构建阶段列表
    phases = []
    prev = min_it
    for b in boundaries:
        if b > prev:
            phases.append((prev, b - 1))
            prev = b
    phases.append((prev, max_it))
    return phases


def plot(rows, output_path, title="AKO-NPU Optimization", hw_info="Ascend 910B"):
    fig, ax = plt.subplots(figsize=(16, 6.5))

    # AKO4ALL 配色
    PHASE_COLORS = [
        '#FFF9C4',  # 淡黄
        '#FFCCBC',  # 淡橙
        '#C8E6C9',  # 淡绿
        '#BBDEFB',  # 淡蓝
        '#F8BBD0',  # 淡粉
        '#D1C4E9',  # 淡紫
        '#FFE0B2',  # 淡橙2
        '#B2DFDB',  # 淡青
    ]

    # 分类
    iters_kept, sp_kept = [], []
    iters_neutral, sp_neutral = [], []
    iters_failed = []
    labels = {}

    for r in rows:
        it, sp, st, lb = r['iter'], r['speedup'], r['status'], r['label']
        if sp is None:
            if st in ('failed', 'reverted'):
                iters_failed.append(it)
            continue
        if st in ('improved', 'baseline'):
            iters_kept.append(it)
            sp_kept.append(sp)
        elif st in ('failed', 'reverted'):
            iters_failed.append(it)
        else:
            iters_neutral.append(it)
            sp_neutral.append(sp)
        if lb and st in ('improved', 'baseline'):
            labels[it] = (sp, lb)

    # Frontier
    frontier_x, frontier_y = [], []
    best = 0
    for r in rows:
        sp = r['speedup']
        if sp is not None:
            if sp > best:
                best = sp
            frontier_x.append(r['iter'])
            frontier_y.append(best)

    if not frontier_x:
        print(f"No data for {output_path}")
        return

    max_sp = max(frontier_y)
    final_sp = frontier_y[-1]
    y_top = max_sp * 1.3

    # 纯色背景
    ax.set_facecolor('#fafafa')

    # ===== Frontier 线 =====
    ax.plot(frontier_x, frontier_y, '-', color='#2d7f2d', linewidth=2.5, zorder=3, label='Best so far (frontier)')

    # ===== 数据点 =====
    ax.scatter(iters_kept, sp_kept, c='#2d7f2d', s=55, zorder=4, edgecolors='white', linewidths=0.5, label='Kept (improvement)')
    if iters_neutral:
        ax.scatter(iters_neutral, sp_neutral, c='#b0b0b0', s=30, zorder=2, alpha=0.6, label='Reverted / neutral')
    if iters_failed:
        fail_y = [max_sp * 0.03] * len(iters_failed)
        ax.scatter(iters_failed, fail_y, c='#cc2222', s=70, marker='x', linewidths=2, zorder=4, label='Failure (incorrect)')

    # ===== Baseline 虚线 =====
    ax.axhline(y=1.0, color='#6699cc', linestyle='--', alpha=0.4, linewidth=1, label='Reference baseline')
    ax.text(frontier_x[-1] + 1, 1.0, 'ref 1.0x', color='#6699cc', fontsize=7, va='center', alpha=0.5)

    # ===== 里程碑标注（最多 6 个，选跳跃最大的）=====
    if labels:
        sorted_labels = sorted(labels.items(), key=lambda x: x[1][0])
        selected = {}
        # 首尾
        if sorted_labels:
            selected[sorted_labels[0][0]] = sorted_labels[0][1]
            selected[sorted_labels[-1][0]] = sorted_labels[-1][1]
        # 中间跳跃大的
        prev_sp = 0
        jumps = []
        for it, (sp, lb) in sorted_labels:
            if sp > prev_sp * 1.12 and it not in selected:
                jumps.append((sp - prev_sp, it, sp, lb))
            prev_sp = max(prev_sp, sp)
        jumps.sort(reverse=True)
        for _, it, sp, lb in jumps[:4]:
            selected[it] = (sp, lb)

        used_y = []
        for it, (sp, lb) in sorted(selected.items()):
            short_lb = lb[:28] if len(lb) > 28 else lb
            # 避免标注重叠
            offset_y = 18
            for uy in used_y:
                if abs(sp - uy) < max_sp * 0.08:
                    offset_y = 30
            used_y.append(sp)

            ax.annotate(f'{short_lb}\n{sp:.2f}x', (it, sp),
                        textcoords="offset points", xytext=(0, offset_y),
                        ha='center', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  edgecolor='#999999', alpha=0.92, linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', color='#999999', lw=0.7))

    # ===== 最终 speedup 醒目框（右上）=====
    ax.annotate(f'{final_sp:.2f}x avg speedup',
                xy=(frontier_x[-1] * 0.75, max_sp * 1.1),
                fontsize=13, fontweight='bold', color='#2d7f2d',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e8',
                          edgecolor='#2d7f2d', linewidth=1.5))

    # ===== 硬件信息框（右下）=====
    n_total = len(rows)
    n_improved = len(iters_kept)
    n_neutral = len(iters_neutral)
    n_failed = len(iters_failed)
    info_text = f'{hw_info}\n{n_total} iters | {n_improved} kept | {n_neutral} neutral | {n_failed} failures'
    ax.text(0.98, 0.05, info_text, transform=ax.transAxes, fontsize=7.5,
            va='bottom', ha='right', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9, linewidth=0.5))

    # ===== 顶部统计行 =====
    stats = f'{n_total} iterations | {n_improved} improvements | 1.0x → {final_sp:.2f}x'
    fig.text(0.5, 0.97, stats, ha='center', fontsize=9.5, color='#444444')

    # ===== 标题 =====
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    # ===== 轴 =====
    ax.set_xlabel('Iteration #', fontsize=11)
    ax.set_ylabel('Average Speedup (x)', fontsize=11)
    ax.set_ylim(0, y_top)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.12, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Input CSV file')
    parser.add_argument('output', help='Output PNG file')
    parser.add_argument('--title', default='AKO-NPU Optimization Trajectory')
    parser.add_argument('--hw', default='Ascend 910B\n48 Vector + 24 Cube cores | GM BW: ~1.8 TB/s')
    args = parser.parse_args()
    rows = load_csv(args.csv)
    plot(rows, args.output, args.title, args.hw)
