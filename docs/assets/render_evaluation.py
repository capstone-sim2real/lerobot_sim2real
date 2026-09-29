"""Create report-style evaluation bar charts with Matplotlib (no robot runtime)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / 'report/최종보고서/evidence/20260914/team_results.json'
report = json.loads(REPORT.read_text())
update = json.loads((HERE / 'evaluation-update.json').read_text())
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(font_path)
plt.rcParams.update({
    'font.family': FontProperties(fname=font_path).get_name(),
    'font.size': 11,
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'svg.hashsalt': 'so101-evaluation',
    'pdf.fonttype': 3,
})

task1 = report['task1']
grasp = update['single_block_grasp']
fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8), sharey=True,
                         gridspec_kw={'width_ratios': [1, 1.45, 1]})
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.27, top=0.80, wspace=0.24)
fig.suptitle('SO-101 실기 평가 결과', fontsize=17, y=0.96)

panels = [
    ('(a) Task 1: 블록 모으기', ['5블록 구역 배치'],
     [100 * task1['mission_successes'] / task1['mission_attempts']],
     ['#4477AA'], [''], '180초 이내 완료 (28/30회)'),
    ('(b) Task 2: 블록 탑쌓기', ['4블록 적층', '5블록 적층'],
     [100 * update['task2']['four_block_stack_success_rate'],
      100 * update['task2']['five_block_stack_success_rate']],
     ['#4477AA', '#4477AA'], ['', '//'], f"블록 수별 적층 성공률 (시행 {update['task2']['attempt_counts']}회)"),
    ('(c) 단일 블록 파지', ['단일 블록 파지'],
     [100 * grasp['successes'] / grasp['attempts']],
     ['#228877'], [''], 'Task 공통 (99/100회)'),
]
for ax, (title, labels, rates, colors, hatches, note) in zip(axes, panels):
    positions = list(range(len(labels)))
    bars = ax.bar(positions, rates, width=0.48, color=colors,
                  edgecolor='#283b48', linewidth=0.7, zorder=3)
    for bar, rate, hatch in zip(bars, rates, hatches):
        bar.set_hatch(hatch)
        value = f'{rate:.1f}'.rstrip('0').rstrip('.') if rate % 1 else str(int(rate))
        ax.text(bar.get_x() + bar.get_width()/2, rate-5, f'{value}%',
                ha='center', va='top', color='white', fontsize=13, fontweight='bold',
                bbox={'facecolor':colors[0], 'edgecolor':'none', 'pad':1.5})
    ax.set_title(title, fontsize=12, pad=16)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.65, len(labels)-0.35)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis='y', color='#dedede', linewidth=0.7, zorder=0)
    ax.tick_params(axis='x', length=0, pad=9)
    ax.tick_params(axis='y', length=3)
    ax.text(0.5, -0.25, note, transform=ax.transAxes, ha='center',
            va='top', fontsize=10, color='#444444')
axes[0].set_ylabel('성공률 (%)', labelpad=10)
fig.text(0.075, 0.065, '자료: Task 1 — 최종보고서 집계 / Task 2 및 단일 파지 — 최신 팀 제공 집계',
         fontsize=9, color='#555555')
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(HERE / f'evaluation-bar-chart.{extension}', dpi=200,
                facecolor='white', metadata={'Creator':'Matplotlib'})
plt.close(fig)
svg = HERE / 'evaluation-bar-chart.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
