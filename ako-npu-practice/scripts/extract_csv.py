#!/usr/bin/env python3
"""
从 ITERATIONS.md 的 Summary 表格提取数据生成标准 CSV。

用法:
    python3 scripts/extract_csv.py <ITERATIONS.md> <output.csv>

支持的表格格式（自动识别）：
  | Iter | Title | Runtime(mean) | Speedup | Status |
  | Iter | Title | Speedup(mean) | Runtime(mean) | Status |
  | Iter | Title | Runtime(ms) | Speedup vs ... | Status |
"""

import re
import sys
import csv


def parse_number(s):
    """从字符串中提取数字，如 '53.41 us' -> 53.41, '1.12x' -> 1.12, '0.198 ms' -> 198.0"""
    s = s.strip().replace('**', '').replace('⭐', '').strip()
    if not s or s == '-' or s == 'N/A':
        return None
    # ms -> us
    m = re.search(r'([\d.]+)\s*ms', s)
    if m:
        return float(m.group(1)) * 1000  # convert to us
    m = re.search(r'([\d.]+)\s*us', s)
    if m:
        return float(m.group(1))
    m = re.search(r'([\d.]+)\s*x', s)
    if m:
        return float(m.group(1))
    m = re.search(r'\b\d+\.?\d*\b', s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def parse_status(s):
    s = s.strip().replace('**', '').lower()
    if 'baseline' in s:
        return 'baseline'
    elif 'improved' in s or 'best' in s:
        return 'improved'
    elif 'fail' in s:
        return 'failed'
    elif 'revert' in s:
        return 'reverted'
    elif 'regression' in s:
        return 'regression'
    elif 'analysis' in s:
        return 'analysis'
    elif 'no-change' in s or 'no change' in s or 'cleanup' in s or 'confirmed' in s:
        return 'no-change'
    else:
        return s.strip()


def extract(md_path):
    rows = []
    with open(md_path) as f:
        lines = f.readlines()

    in_table = False
    header_cols = []

    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            if in_table:
                break  # table ended
            continue

        cells = [c.strip() for c in line.split('|')[1:-1]]

        # Skip separator line
        if all(set(c) <= set('-: ') for c in cells):
            in_table = True
            continue

        # Detect header
        if not in_table and any('iter' in c.lower() for c in cells):
            header_cols = [c.lower() for c in cells]
            in_table = True
            continue

        if not in_table:
            continue

        if len(cells) < 3:
            continue

        # Parse iter number
        iter_str = cells[0].strip().replace('**', '')
        m = re.match(r'(\d+)', iter_str)
        if not m:
            continue
        iter_num = int(m.group(1))

        title = cells[1].strip().replace('**', '').replace('⭐', '').strip()

        # Find runtime and speedup based on header or heuristics
        runtime_us = None
        speedup = None
        status = ''

        for i, cell in enumerate(cells[2:], 2):
            val = parse_number(cell)
            h = header_cols[i] if i < len(header_cols) else ''

            if 'speedup' in h:
                speedup = val
            elif 'runtime' in h or 'duration' in h:
                runtime_us = val
            elif 'status' in h:
                status = parse_status(cell)
            elif val is not None and 'x' in cell and speedup is None:
                speedup = val
            elif val is not None and ('us' in cell or 'ms' in cell) and runtime_us is None:
                runtime_us = val
            elif i == len(cells) - 1 and not status:
                status = parse_status(cell)

        # Label: only for key milestones
        label = ''
        if status in ('baseline', 'improved') or '⭐' in cells[1] or '**best**' in line.lower():
            label = title[:40]

        rows.append({
            'iter': iter_num,
            'runtime_us': runtime_us if runtime_us else '',
            'speedup': speedup if speedup else '',
            'status': status,
            'label': label,
        })

    return rows


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ITERATIONS.md> <output.csv>")
        sys.exit(1)

    rows = extract(sys.argv[1])
    with open(sys.argv[2], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iter', 'runtime_us', 'speedup', 'status', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} rows -> {sys.argv[2]}")
