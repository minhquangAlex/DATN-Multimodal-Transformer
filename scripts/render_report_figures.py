"""Render report figures from recorded Stage 1 metadata and measured results."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/figures'
SUMMARY = ROOT / 'processed_data/v2/stage1_summary.json'
RESULTS = ROOT / 'results/v2/20260916T150005Z_ea67e16e/baseline_results.csv'
NAVY, BLUE, GREEN, AMBER, GRAY = '#172B4D', '#2563EB', '#13876D', '#C77715', '#526277'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': NAVY, 'axes.labelcolor': GRAY,
                     'xtick.color': GRAY, 'ytick.color': GRAY,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})


def save(fig, name):
    for extension in ['png', 'svg']:
        fig.savefig(OUTPUT / f'{name}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def render():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    results = pd.read_csv(RESULTS)
    run = json.loads(RESULTS.with_name('run_config.json').read_text(encoding='utf-8'))
    for name in ['selected_articles', 'dataset_index']:
        if run['data_hashes'][f'{name}.csv'] != summary[f'{name}_sha256']:
            raise ValueError('Recorded baseline run and current Stage 1 data do not match')
    if set(results.Model) != {'zero', 'last_value', 'moving_average'} or len(results) != 3:
        raise ValueError('Expected the three recorded deterministic baselines')
    fig, ax = plt.subplots(figsize=(14, 5.8))
    fig.subplots_adjust(left=.06, right=.96, bottom=.15, top=.72)
    fig.text(.06, .91, 'CHIA DỮ LIỆU THEO THỜI GIAN', fontsize=21, weight='bold')
    fig.text(.06, .85, f"{summary['n_selected_articles']:,} sản phẩm  •  {summary['n_weeks']} tuần đầy đủ  •  Chọn sản phẩm chỉ từ train", color=GRAY)
    start = 0
    for key, title, color in [('train', 'Train', BLUE), ('val', 'Validation', GREEN), ('test', 'Test', AMBER)]:
        n = summary['splits'][key]['n_weeks']
        ax.add_patch(Rectangle((start, .62), n, .28, facecolor=color, edgecolor='white', linewidth=2))
        ax.text(start+n/2, .76, f'{title}\n{n} tuần', ha='center', va='center', color='white', fontsize=11, weight='bold')
        start += n
    ticks = [0, summary['splits']['train']['n_weeks'],
             summary['splits']['train']['n_weeks'] + summary['splits']['val']['n_weeks'], start]
    ax.set(xlim=(0, start), ylim=(0, 1), yticks=[], xticks=ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_xlabel('Số tuần tích lũy từ 2018-W39', labelpad=10)
    ax.spines['left'].set_visible(False)
    for x, name, dates, count, color in [(.06, 'TRAIN', '2018-W39 → 2020-W14', 536000, BLUE),
                                       (.38, 'VALIDATION', '2020-W15 → 2020-W26', 88000, GREEN),
                                       (.70, 'TEST', '2020-W27 → 2020-W38', 88000, AMBER)]:
        fig.text(x, .40, name, color=color, weight='bold', fontsize=12)
        fig.text(x, .34, dates, fontsize=11)
        fig.text(x, .28, f'{count:,} cửa sổ với L=12, H=2', color=GRAY, fontsize=11)
    fig.text(.06, .025, 'Loại hai tuần biên không đủ 7 ngày. Mọi tuần target trong một cửa sổ phải thuộc cùng một tập.', fontsize=10, color=GRAY)
    save(fig, 'time_split_v2')

    groups = pd.Series(summary['group_distribution']).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=.28, right=.92, top=.80, bottom=.15)
    fig.text(.06, .92, 'PHÂN BỐ SẢN PHẨM SAU LẤY MẪU', fontsize=21, weight='bold')
    fig.text(.06, .86, f"{len(groups)} nhóm  •  Tổng {groups.sum():,} sản phẩm  •  Tiêu chí dựa trên tập train", color=GRAY)
    bars = ax.barh(groups.index, groups.values, color=BLUE, height=.65)
    ax.bar_label(bars, labels=[f'{value:,}' for value in groups], padding=6, color=NAVY, fontsize=10)
    ax.set_xlim(0, groups.max()*1.16)
    ax.set_xlabel('Số sản phẩm')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E5EBF2', linewidth=.8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    fig.text(.06, .035, 'Nguồn: stage1_summary.json v2. Nhãn nhóm giữ nguyên theo bộ dữ liệu H&M; nhóm nhỏ vẫn được giữ trong mẫu.', fontsize=10, color=GRAY)
    save(fig, 'product_groups_v2')

    ordered = results.set_index('Model').loc[['zero', 'last_value', 'moving_average']]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.95, top=.78, bottom=.14, hspace=.60, wspace=.28)
    fig.text(.06, .93, 'KẾT QUẢ BASELINE ĐÃ CHẠY TRÊN TEST THẬT', fontsize=20, weight='bold')
    fig.text(.06, .87, '88.000 cửa sổ  •  H=2  •  176.000 giá trị dự báo  •  Giá trị thấp hơn là tốt hơn', color=GRAY)
    for ax, metric in zip(axes.flat, ['MAE', 'RMSE', 'wMAPE (%)', 'sMAPE (%)']):
        vals = ordered[metric]
        bars = ax.bar(['Zero', 'Last-value', 'Moving Avg. 4'], vals, color=['#97A6BA', GREEN, BLUE], width=.58)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in vals], padding=4, fontsize=10)
        ax.set_ylim(0, vals.max()*1.25)
        ax.set_title(metric, loc='left', fontsize=13, weight='bold', pad=10)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E5EBF2', linewidth=.8)
    fig.text(.06, .045, 'Chỉ gồm ba baseline không cần huấn luyện. Chưa có kết quả deep learning hoặc Multimodal Transformer trong hình này.', fontsize=10, color=GRAY)
    fig.text(.06, .02, 'Nguồn: run 20260916T150005Z_ea67e16e. Đây chưa phải bảng kết quả cuối cùng của đồ án.', fontsize=10, color=GRAY)
    save(fig, 'baseline_results_v2')

    manifest = {'sources': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [SUMMARY, RESULTS]},
                'figures': ['time_split_v2', 'product_groups_v2', 'baseline_results_v2'],
                'formats': ['png', 'svg'], 'status': 'recorded data, not simulated results',
                'window_config': {'seq_len': 12, 'pred_len': 2}}
    (OUTPUT / 'figure_sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    render()
