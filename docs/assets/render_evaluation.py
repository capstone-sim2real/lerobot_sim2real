"""Render README evaluation cards from the report and the latest team update."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'report/최종보고서/evidence/20260914/team_results.json'
data = json.loads(DATA.read_text())
update = json.loads((HERE / 'evaluation-update.json').read_text())
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
bold = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
fig = plt.figure(figsize=(12, 7.3), facecolor='#f3f6fa')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1200), ylim=(0, 730)); ax.axis('off')

def text(x, y, value, size=13, color='#172b42', weight=False, **kw):
    ax.text(x, y, value, fontsize=size, color=color,
            fontproperties=bold if weight else font, va='center', **kw)

def rect(x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle=f'round,pad=0,rounding_size={min(14, h/2)}',
                              facecolor=color, edgecolor='none'))

text(42, 680, 'SO-101  /  실기 평가', 24, weight=True)
text(42, 640, 'Task 1: 최종보고서 집계  /  Task 2·파지: 최신 팀 집계', 12, '#5c6f83')
rect(34, 205, 554, 390, '#ffffff'); rect(612, 205, 554, 390, '#ffffff')
t1, t2 = data['task1'], data['task2']
n, success = t1['mission_attempts'], t1['mission_successes']
text(62, 555, 'TASK 1  ·  블록 모으기', 17, weight=True)
text(62, 515, '180초 안에 블록 5개를 지정 구역에 배치', 12, '#5c6f83')
text(62, 439, f'{success/n:.1%}', 48, '#008975', True)
text(62, 384, f'{n}회 중 {success}회 완료', 16, weight=True)
rect(62, 318, 490, 25, '#fae4e3')
rect(62, 318, 490*success/n, 25, '#11a58b')
text(62, 291, f'성공 {success}회', 12, '#008975')
text(552, 291, f'실패 {t1["mission_failures"]}회', 12, '#bc5650', ha='right')
text(62, 246, '실패 원인: 파지 중 블록이 사거리 밖으로 밀려 시간 초과', 10, '#5c6f83')
text(640, 555, 'TASK 2  ·  블록 탑쌓기', 17, weight=True)
text(640, 515, '블록 수별 적층 성공률', 12, '#5c6f83')
for label, rate, y in [
    ('5블록 적층', update['task2']['five_block_stack_success_rate'], 425),
    ('4블록 적층', update['task2']['four_block_stack_success_rate'], 325),
]:
    text(640, y+25, label, 15, weight=True)
    text(1136, y+25, f'{rate:.0%}', 24, '#376bd7', True, ha='right')
    rect(640, y-15, 496, 20, '#edf1f7')
    rect(640, y-15, 496*rate, 20, '#4e81ed')
text(640, 238, '팀 제공 성공률 · 시행 횟수 및 5초 유지 조건 미제공', 11, '#5c6f83')
rect(34, 56, 1132, 127, '#e5ecf5')
text(62, 148, '단일 블록 파지 성공률', 16, weight=True)
text(62, 105, 'Task 구분 없이 집계', 12, '#5c6f83')
grasp = update['single_block_grasp']
text(700, 116, f"{grasp['successes']/grasp['attempts']:.0%}", 36, '#008975', True)
text(865, 116, f"{grasp['attempts']}회 중 {grasp['successes']}회 성공", 17, weight=True)
text(42, 27, '출처: 최종보고서 team_results.json + README 갱신 시 제공한 팀 평가 집계', 10, '#5c6f83')
fig.savefig(HERE/'evaluation-results.png', dpi=160, facecolor=fig.get_facecolor())
fig.savefig(HERE/'evaluation-results.svg', facecolor=fig.get_facecolor())
plt.close(fig)

svg = HERE / 'evaluation-results.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
