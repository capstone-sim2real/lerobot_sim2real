"""Render README evaluation cards from the preserved report aggregate."""
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
text(42, 640, '최종보고서 팀 집계 · 집계 확인 2026.09.14', 12, '#5c6f83')
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
text(640, 515, '같은 20회 시행에서 도달한 최대 높이 · 누적', 12, '#5c6f83')
for level, y, color in [('3',443,'#4e81ed'), ('4',369,'#4e81ed'), ('5',295,'#a8b6ca')]:
    count = t2['cumulative_height_reach_counts'][level]
    total = t2['mission_attempts']
    text(640,y+19,f'{level}단 이상',13,weight=True)
    text(1136,y+19,f'{count/total:.0%}  ·  {count}/{total}회',12,ha='right')
    rect(640,y-15,496,16,'#edf1f7')
    if count: rect(640,y-15,496*count/total,16,color)
text(640, 238, '5단 도달 0회 · 종료 시 높이 / 5초 유지 여부 미집계', 11, '#5c6f83')
rect(34, 56, 1132, 127, '#e5ecf5')
text(62, 148, '파지 성공률', 14, weight=True)
text(62, 111, '재시도를 포함한 전체 파지 시도 기준', 11, '#5c6f83')
for name, item, x in [('Task 1', t1, 485), ('Task 2', t2, 830)]:
    good, total = item['grasp_successes'], item['grasp_attempts_including_retries']
    text(x,146,name,12,'#5c6f83',True)
    text(x,103,f'{good/total:.1%}',25,weight=True)
    text(x+125,103,f'{good}/{total}회',12,'#5c6f83')
text(42, 27, '출처: 최종보고서 team_results.json  ·  현재 커밋 재평가 결과가 아닌 보고서 집계', 10, '#5c6f83')
fig.savefig(HERE/'evaluation-results.png', dpi=160, facecolor=fig.get_facecolor())
fig.savefig(HERE/'evaluation-results.svg', facecolor=fig.get_facecolor())
plt.close(fig)

svg = HERE / 'evaluation-results.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
