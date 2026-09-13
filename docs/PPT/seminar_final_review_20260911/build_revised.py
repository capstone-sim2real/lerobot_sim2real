"""Rebuild editable seminar slides while retaining original embedded videos.

Run: uv run --no-project --with python-pptx --with pillow python build_revised.py
"""
from pathlib import Path
import hashlib
import json
from io import BytesIO

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/ehdrms/Downloads/졸업과제_세미나_발표자료__최종_20260911.pptx')
DEST = ROOT / '졸업과제_세미나_발표자료_수정본_20260911.pptx'
A = ROOT / 'assets'
F = ROOT / 'analysis'
FONT = 'Noto Sans CJK KR'
C = dict(bg='F7F8F0', navy='355872', blue='7AAACE', pale='E6F0F6', sky='9CD5FF',
         text='243541', muted='5B6B75', line='D6E1E5', white='FFFFFF', amber='936744')
ORDER = [1, 2, 7, 8, 5, 6, 9, 10, 11, 3, 4, 13, 12, 14]
DURATIONS = [15, 35, 40, 45, 35, 20, 50, 45, 40, 45, 40, 40, 45, 20]
prs = Presentation(SOURCE)
prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
slides = list(prs.slides)
text_inventory = []


def color(v):
    return RGBColor.from_string(C.get(v, v))


def rect(s, x, y, w, h, fill='white', line=None, radius=False):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                            Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color(fill)
    if line:
        sh.line.color.rgb = color(line); sh.line.width = Pt(.7)
    else:
        sh.line.fill.background()
    if radius:
        sh.adjustments[0] = .08
    return sh


def txt(s, x, y, w, h, text, size=18, c='text', bold=False, align=PP_ALIGN.LEFT):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.clear(); tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = 1.13; p.space_after = Pt(5)
        r = p.add_run(); r.text = line
        r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color(c)
        rp = r._r.get_or_add_rPr(); rp.set('lang', 'ko-KR')
        for tag in ['a:ea', 'a:cs']:
            e = rp.find(qn(tag))
            if e is None:
                e = OxmlElement(tag); rp.append(e)
            e.set('typeface', FONT)
    text_inventory.append({'slide_original': slides.index(s)+1, 'text': text, 'size_pt':size,
                           'box_in':[x,y,w,h]})
    return sh


def line(s, x1, y1, x2, y2, c='line', width=1, arrow=False):
    sh = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb = color(c); sh.line.width = Pt(width)
    if arrow:
        tail = OxmlElement('a:tailEnd'); tail.set('type','triangle')
        sh.line._get_or_add_ln().append(tail)
    return sh


def photo(s, path, x, y, w, h):
    # Contain the complete original image; no raster retouching or invented pixels.
    iw, ih = Image.open(path).size
    scale = min(w/iw, h/ih)
    pw, ph = iw*scale, ih*scale
    return s.shapes.add_picture(str(path), Inches(x+(w-pw)/2), Inches(y+(h-ph)/2),
                               width=Inches(pw), height=Inches(ph))


def panel(s, x, y, w, h, title, body, size=18):
    rect(s,x,y,w,h)
    line(s,x,y,x+w,y,'blue',2)
    txt(s,x+.2,y+.18,w-.4,.4,title,18,'navy',True)
    txt(s,x+.2,y+.68,w-.4,h-.8,body,size)


def clear(s):
    for sh in list(s.shapes):
        if '<a:videoFile' not in sh._element.xml:
            s.shapes._spTree.remove(sh._element)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = color('bg')


def header(s, section, title, subtitle, num):
    txt(s,.55,.24,11,.21,section,10,'muted')
    txt(s,.55,.64,12.2,.58,title,27,'navy',True)
    txt(s,.55,1.3,12.2,.38,subtitle,14,'muted')
    line(s,.55,1.88,12.78,1.88)
    line(s,.55,7.02,12.78,7.02)
    txt(s,.55,7.15,10,.18,'PUSAN NATIONAL UNIVERSITY  /  Team Sim2Real',9,'muted')
    txt(s,12.15,7.1,.63,.27,f'{num:02d} / 14',10,'muted',align=PP_ALIGN.RIGHT)


def note(s, title, body, orig):
    new = ORDER.index(orig)+1
    if s.notes_slide.notes_text_frame is None:
        ns = s.notes_slide
        sid = max((sh.shape_id for sh in ns.shapes), default=0)+1
        body = parse_xml(f'''<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:nvSpPr><p:cNvPr id="{sid}" name="발표자 노트"/><p:cNvSpPr/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>''')
        ns.shapes._spTree.append(body)
    s.notes_slide.notes_text_frame.text = f'{new:02d}. {title}\n목표 {DURATIONS[new-1]}초\n\n{body}'


def movie(s, x, y, w, poster):
    sh = next(sh for sh in s.shapes if '<a:videoFile' in sh._element.xml)
    sh.left, sh.top, sh.width, sh.height = Inches(x), Inches(y), Inches(w), Inches(w*9/16)
    # Keep media, original playback actions and timing. Change only poster and bounds.
    _, rid = s.part.get_or_add_image_part(str(poster))
    blip = sh._element.find('.//' + qn('a:blip'))
    blip.set(qn('r:embed'), rid)
    compatible = A / 'media3_h264.mp4'
    if s is slides[12] and compatible.is_file():
        video_ref = sh._element.find('.//' + qn('a:videoFile')).get(qn('r:link'))
        s.part.related_part(video_ref)._blob = compatible.read_bytes()
    return sh


for s in slides:
    clear(s)

# 01 / original 01: formal title, small logo, restrained palette.
s=slides[0]
line(s,.7,.69,12.63,.69,'blue',1.3)
photo(s,A/'image3.png',11.1,1.05,1.42,1.42)
txt(s,.75,1.58,9.8,.35,'부산대학교 졸업과제 세미나',16,'muted')
txt(s,.75,2.46,11.6,1.5,'컴퓨터 비전과 역기구학 기반 SO-ARM101\n로봇팔의 블록 이동 및 적재 시스템',29,'navy',True)
txt(s,.78,4.38,10,.42,'Team Sim2Real',21,'navy',True)
txt(s,.78,4.99,10,.38,'윤민석 / 김주환 / 이동근',17,'muted')
line(s,.78,6.23,12.56,6.23)
txt(s,.78,6.5,11.8,.36,'블록 인식  /  좌표 변환  /  파지 확인  /  미션 실행',15,'muted')
note(s,'프로젝트 소개','고정 카메라의 컴퓨터 비전과 역기구학, FSM을 연결한 블록 조작 시스템입니다. 이동과 적재의 실제 동작, 설계 선택, 남은 검증을 설명합니다.',1)

# 02 / original 02: requirements, not tuning constants.
s=slides[1]
header(s,'01  PROJECT','프로젝트 목표와 작업 환경','같은 규격의 블록 5개를 고정 카메라와 로봇팔로 조작',2)
rect(s,.55,2.12,5.78,4.61)
photo(s,A/'image4.jpeg',.77,2.25,5.34,3.75)
txt(s,.84,6.14,5.12,.35,'시스템 구성 개념도',12,'muted')
panel(s,6.62,2.12,6.16,1.93,'1차 미션  /  지정 영역 배치','180초 안에 20 × 10 cm 영역으로 운반\n경계 2 cm 걸침 허용, 적재는 불인정',18)
panel(s,6.62,4.28,6.16,1.9,'2차 미션  /  블록 적재','300초 안에 적재 후 5초 이상 유지\n동일한 블록 5개로 수행',18)
txt(s,6.8,6.41,5.8,.33,'공통 흐름: 인식 → 파지 → 확인 → 운반 → 배치',14,'navy')
note(s,'미션 규격','1차는 180초 안에 20×10cm 영역으로 운반하고, 2차는 300초 안에 적재해 5초 이상 유지하는 과제입니다. 왼쪽은 구성 개념도이며 실제 장비 사진이 아닙니다. 카메라 한 대와 로봇 베이스 좌표를 공통으로 사용합니다.',2)

# 03 / original 07: a development decision, not an unperformed benchmark.
s=slides[6]
header(s,'02  APPROACH','모방학습에서 CV + IK로 전환','고정된 작업 조건을 활용하고, 실패를 단계별로 확인할 수 있는 구조를 선택',3)
panel(s,.55,2.12,3.88,3.42,'초기 ACT 실험','단일 블록 이동\n50회 중 29회 성공\n성공률 58%',21)
txt(s,.78,4.61,3.45,.7,'초기 실험 조건의 결과\n최종 방식과 동일 조건 비교 없음',13,'muted')
panel(s,4.7,2.12,3.88,3.42,'확장 과정의 어려움','새 위치와 시야 변화에 대응\n데이터 수집과 재학습 부담\n접촉 실패의 원인 분리 필요',18)
panel(s,8.85,2.12,3.93,3.42,'최종 CV + IK','색과 형상으로 위치 검출\n기구학으로 자세 계산\n센서와 FSM으로 실패 대응',18)
rect(s,.55,5.86,12.23,.87,'pale')
txt(s,.8,6.08,11.74,.45,'전환의 근거: 고정 평면과 규격 블록이라는 조건, 실패를 추적하고 수정할 수 있는 구조',18,'navy',True)
note(s,'접근법 전환','초기 ACT는 단일 블록 이동 50회 중 29회 성공했습니다. ACT와 CV/IK를 동일 조건에서 반복 비교한 결과는 없으므로 성공률 상승을 주장하지 않습니다. 하이브리드 방식을 검토했지만 접촉 실패 분석과 데이터 수집 부담이 남았습니다. 최종적으로 알려진 작업 조건을 이용해 인식, 좌표, 계획, 접촉을 분리했습니다. SmolVLA도 초기 시도에 포함되지만 안정적인 파지 성능 수치는 확보하지 못했습니다.',7)

# 04 / original 08: show both data flow and FSM recovery.
s=slides[7]
header(s,'03  ARCHITECTURE','인식에서 실행까지 연결한 시스템','카메라 / OpenCV / PlaCo 기반 기구학 / LeRobot 입출력 / FSM',4)
steps=[('카메라 영상','최신 프레임'),('평면 보정 / 검출','블록 XY와 각도'),('파지 후보 / IK','실행할 관절 자세'),('관절 이동 / 센싱','명령과 측정 상태')]
for j,(t,b) in enumerate(steps):
    x=.55+j*3.15
    rect(s,x,2.17,2.78,1.12,'white')
    txt(s,x+.16,2.34,2.46,.38,t,17,'navy',True)
    txt(s,x+.16,2.82,2.46,.3,b,13,'muted')
    if j<3: txt(s,x+2.82,2.47,.3,.35,'→',20,'blue')
txt(s,.58,3.61,12,.34,'FSM  /  파지가 확인된 경우에만 운반으로 전이',18,'navy',True)
states=[('SELECT','대상 선택'),('PICK','후보 파지'),('VERIFY','파지 확인'),('TRANSPORT','운반'),('PLACE','배치')]
for j,(t,b) in enumerate(states):
    x=.55+j*2.53
    rect(s,x,4.25,2.1,1.02,'navy' if j==2 else 'white')
    txt(s,x+.1,4.38,1.9,.32,t,15,'white' if j==2 else 'navy',True,PP_ALIGN.CENTER)
    txt(s,x+.1,4.82,1.9,.25,b,12,'white' if j==2 else 'muted',False,PP_ALIGN.CENTER)
    if j<4: txt(s,x+2.16,4.57,.35,.35,'→',19,'blue')
line(s,6.66,5.32,6.66,5.72,'blue',1.5)
line(s,6.66,5.72,1.59,5.72,'blue',1.5)
line(s,1.59,5.72,1.59,5.32,'blue',1.5,True)
txt(s,2.27,5.88,7.8,.32,'미확인 → 후보 재시도 또는 대상 재선택',15,'navy')
txt(s,.58,6.48,12,.32,'공통 파지 흐름에 지정 영역 배치 / 적재 전략을 연결',16,'muted')
note(s,'시스템과 FSM','카메라 영상을 호모그래피로 mm 기준 평면에 보정한 뒤 블록을 검출합니다. 후보별 IK를 계산하고 LeRobot 입출력으로 실행합니다. MoveIt은 사용하지 않고 PlaCo 기반 기구학과 직접 구현한 관절 보간을 연결했습니다. PICK 내부에서 다른 후보를 시도하고, 실패하면 SELECT로 돌아갑니다. VERIFY가 확인되기 전에는 TRANSPORT로 가지 않습니다. 웹 뷰어는 관찰용 가지이며 화면 오버레이가 제어 입력으로 돌아가지는 않습니다.',8)

# 05 / original 05: image plus legible explanations.
s=slides[4]
header(s,'04  PERCEPTION','색과 형상으로 블록을 검출','색이 같은 경계 테이프를 구분하고, 실제 작업 대상만 남기는 과정',5)
rect(s,.55,2.12,7.63,4.61)
photo(s,A/'image9.jpeg',.73,2.3,7.27,3.98)
txt(s,.84,6.36,6.98,.26,'실제 검출 화면 / 블록 후보, 방향, 지정 영역 표시',12,'muted')
for y,t,b in [(2.12,'01  색 후보','HSV 범위로 후보 영역 생성'),(3.72,'02  형상 확인','면적, 종횡비, 채움 정도 검사'),(5.32,'03  작업 대상 선택','영역 안 / 작업 범위 밖 제외')]:
    panel(s,8.47,y,4.31,1.41,t,b,16)
note(s,'색과 형상 검출','빨간 경계 테이프와 블록의 색이 겹치므로 색만으로 판단하지 않습니다. 알려진 블록 크기와 형상으로 다시 검사하고 이미 지정 영역 안에 있는 블록은 제외합니다. 세부 임계값은 9월 8일 기준 면적 900–3200mm², 종횡비 1.6 이하, fill 0.65 이상, solidity 0.78 이상입니다. 적용 전후 정확도 개선율은 측정하지 않았습니다.',5)

# 06 / original 06: preserve embedded monitoring video.
s=slides[5]
header(s,'05  MONITORING','검출 결과와 파지 후보의 실시간 관찰','영상과 측정 상태를 함께 보여 주는 카메라 웹 뷰어',6)
movie(s,.55,2.14,8.42,F/'media2_070.jpg')
panel(s,9.24,2.14,3.54,1.27,'검출 결과','블록 중심과 각도',16)
panel(s,9.24,3.61,3.54,1.27,'파지 후보','기본점과 보정 후보',16)
panel(s,9.24,5.08,3.54,1.65,'FK 표시의 의미','관절값으로 계산한 위치\n실제 턱의 영상 추적과 구분',15)
note(s,'카메라 웹 뷰어','선택 재생 10–15초. 원본 영상은 약 106초이므로 전부 재생하지 않습니다. 검출 결과, 제외 영역, 후보 좌표를 보여 주는 기능을 설명합니다. FK 표시는 관절값으로 계산한 모델 기준점의 투영이며 영상에서 실제 그리퍼 턱을 검출한 위치가 아닙니다. 영상 원본을 내장했고 클릭 재생 동작을 유지했습니다.',6)

# 07 / original 09: one calibration set, no misleading improvement arrow.
s=slides[8]
header(s,'06  CALIBRATION','영상 좌표와 로봇 좌표를 대응시키는 과정','블록 윗면에서 수집한 대응점으로 호모그래피 H를 추정',7)
rect(s,.55,2.12,5.87,4.61)
photo(s,A/'image13.png',.74,2.29,5.49,3.09)
txt(s,.83,5.54,5.25,.3,'블록 윗면에서 재수집한 9개 대응점',13,'muted')
txt(s,.84,6.07,2.62,.4,'RMS  8.18 mm',19,'navy',True)
txt(s,3.36,6.1,2.76,.4,'최대 LOO  26.64 mm',16,'navy',True)
for y,t,b in [(2.12,'01  로봇 좌표 기록','윗면 중앙에 TCP 정렬\n관절값 → FK → 베이스 기준 XY'),(3.64,'02  영상 좌표 기록','블록은 그대로, 팔만 이동\n같은 윗면 중앙의 픽셀 좌표 수집'),(5.16,'03  대응쌍으로 H 추정','(u, v) → H → (x, y) mm\n고정 카메라 / 블록 윗면 평면')]:
    panel(s,6.71,y,6.07,1.42,t,b,16)
note(s,'캘리브레이션','TCP를 블록 윗면 중앙에 맞춘 상태에서 관절값과 FK를 기록하고, 블록을 그대로 둔 채 팔을 치워 같은 점의 픽셀 좌표를 기록합니다. 한 물리적 점을 두 좌표로 연결한 것이 대응쌍입니다. 9월 8일 재수집 9점의 RMS는 8.18mm, 최대 LOO는 26.64mm로 개발 목표 RMS 5mm와 최대 LOO 8mm에 미달합니다. RMS는 적합에 사용한 점의 잔차이고 LOO는 한 점을 제외하고 예측한 오차입니다. 실제 턱 위치를 독립 측정한 오차는 아닙니다. 15점과 9점은 수집 조건이 다르므로 점 수 감소의 개선 효과로 해석하지 않습니다.',9)

# 08 / original 10: native schematic, no generated result image.
s=slides[9]
header(s,'07  KINEMATICS','중심점 실패에도 실행 가능한 파지 후보 탐색','후보별 IK 도달성을 확인하고, 실제 파지는 센서로 다시 판단',8)
rect(s,.55,2.12,5.61,4.61)
txt(s,.81,2.32,5.09,.37,'기본점과 전 / 후 / 좌 / 우 후보',18,'navy',True)
cx,cy=3.35,4.68
line(s,cx-1.3,cy,cx+1.3,cy,'blue',1.3)
line(s,cx,cy-1.15,cx,cy+1.15,'blue',1.3)
rect(s,cx-.32,cy-.32,.64,.64,'pale','blue')
points=[(cx,cy,'기본',-.18,.48),(cx,cy-1.1,'전',.22,-.15),(cx,cy+1.1,'후',.22,-.12),(cx-1.2,cy,'우',-.48,-.14),(cx+1.2,cy,'좌',.18,-.14)]
for x,y,label,dx,dy in points:
    sh=s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(x-.065),Inches(y-.065),Inches(.13),Inches(.13))
    sh.fill.solid();sh.fill.fore_color.rgb=color('navy');sh.line.fill.background()
    txt(s,x+dx,y+dy,.65,.32,label,14,'navy',True)
txt(s,.84,3.01,4.99,.38,'반경 / 접선 방향의 15 mm 오프셋',15,'muted')
txt(s,.83,6.23,5.05,.27,'후보 배치 개념도 / 실제 성공 위치를 뜻하지 않음',11,'muted')
panel(s,6.45,2.12,6.33,1.3,'후보 계산','중심이 실패해도 도달 가능한 후보가 있으면 실행',16)
panel(s,6.45,3.62,6.33,1.3,'실행 후 검증','위치와 부하로 파지 확인, 미확인 시 다음 후보',16)
panel(s,6.45,5.12,6.33,1.61,'중립 손목 정렬','정사각형의 90° 대칭 중 중립에 가까운 방향 선택\nIK 오차는 모델 내부의 목표 도달 오차',16)
note(s,'IK와 후보 재시도','IK는 목표 위치와 방향을 만족하는 관절 자세를 계산합니다. 기본 중심 후보가 실패해도 실행 가능한 다른 후보가 있으면 계속 시도하도록 보완했습니다. 전후좌우는 그리퍼 기준 반경 및 접선 방향입니다. 후보 수와 재시도 횟수는 구분하며 최대 4회로 단정하지 않습니다. 320mm는 사용한 관측 범위 설정이며 로봇의 모든 자세에 대한 절대 도달 한계가 아닙니다. 70mm 턱 간격을 독립 실측한 근거가 없어 15mm 오차를 구조적으로 보장한다는 표현은 삭제했습니다.',10)

# 09 / original 11: readable engineering choices, correct software layer.
s=slides[10]
header(s,'08  ENGINEERING','실기에서 드러난 문제와 설계 대응','검출, 자세, 접촉의 문제를 나누어 제어 조건에 반영',9)
cards=[('색상 간섭','테이프와 블록의 색 중복','색 후보에 형상 조건 추가\n면적 / 종횡비 / 채움 정도'),('손목 서보 과열','고정 방향 유지 중 손목 비틀림','중립에 가까운 방향 선택\n블록의 90° 대칭 활용'),('빈손 운반 위험','목표 도달과 실제 파지는 다름','그리퍼 위치와 부하 확인\n미확인 시 운반 전이 금지')]
for j,(t,problem,solution) in enumerate(cards):
    x=.55+j*4.16
    rect(s,x,2.12,3.91,3.44)
    txt(s,x+.2,2.35,3.51,.4,t,21,'navy',True)
    txt(s,x+.2,3.1,3.51,.68,problem,17,'muted')
    line(s,x+.2,3.92,x+3.71,3.92)
    txt(s,x+.2,4.2,3.51,1.0,solution,17)
rect(s,.55,5.84,12.23,.89,'pale')
txt(s,.81,6.05,11.71,.47,'측정 자세에서 시작 → 관절 변화량 제한 → LeRobot 입출력 클램프 → 파지 확인',17,'navy',True)
note(s,'문제와 설계 대응','세 가지 문제를 각각 형상 조건, 중립 손목 정렬, 위치와 부하를 결합한 파지 판정으로 다뤘습니다. 원본의 125도는 한계각 관련 값, 3.7도는 모델 기반 결과여서 같은 조건의 실측 개선 그래프로 제시하지 않습니다. 과열 재발 없음과 센서 오분류 없음도 반복 검증 결과로 단정하지 않습니다. max_relative_target은 로봇 펌웨어 자체가 아니라 LeRobot 입출력 계층의 상대 목표 제한입니다. 9월 8일 기준 파지 임계값은 위치 12, 절대 부하 200이며 새 영상의 설정값과 자동으로 동일하다고 가정하지 않습니다.',11)

# 10 / original 03: distinguish the logged trial from the submitted demo.
s=slides[2]
header(s,'09  RESULTS','1차 미션: 5개 블록의 운반과 배치','블록 검출부터 다음 대상의 재선택까지 연속 동작으로 연결',10)
panel(s,.55,2.12,5.86,2.0,'실제 동작에서 확인한 결과','블록 5개를 지정 영역으로 운반\n파지와 배치를 순차적으로 반복',21)
panel(s,.55,4.38,5.86,1.31,'실패 대응','파지 미확인 시 다른 후보 또는 대상 재선택',16)
rect(s,.55,6.03,5.86,.7,'pale')
txt(s,.76,6.2,5.43,.36,'추가 평가: 반복 성공률과 미션 완료시간',16,'navy',True)
rect(s,6.69,2.12,6.09,4.61)
photo(s,F/'media1_103.jpg',6.87,2.33,5.73,3.24)
txt(s,6.92,5.79,5.59,.39,'실제 동작 영상의 최종 배치 장면',17,'navy',True)
txt(s,6.92,6.31,5.59,.3,'파지 → 운반 → 배치 → 다음 블록 선택',14,'muted')
note(s,'1차 미션 결과','이번 PPT 내장 영상의 103초 프레임에서 5개 블록이 지정 영역에 모인 장면을 확인했습니다. 사용자는 날짜 대신 전체 성과 중심으로 발표를 구성하도록 요청해 본문에서 날짜와 기록 대조 설명을 뺐습니다. 참고로 기존 보고서의 9월 8일 외부 180초 제한 시험은 후보 실행 14회, HELD 4회, EMPTY 9회, 판정 전 중단 1회, 슬롯 해제 4회였으며 마지막 블록이 남았습니다. 이 기록과 이번 영상을 동일 시행으로 합치지 않습니다. 촬영 날짜와 배속 및 편집 여부를 확인하기 전까지 영상 길이 105.7초를 공식 완료시간으로 사용하지 않습니다. 성공 장면을 반복 성공률로 환산하지 않습니다.',3)

# 11 / original 04: preserved video with short cues.
s=slides[3]
header(s,'10  DEMONSTRATION','1차 미션 동작 영상','블록을 선택하고, 파지한 뒤, 지정 영역으로 운반하는 과정',11)
movie(s,.55,2.14,8.42,F/'media1_103.jpg')
for y,t,b in [(2.14,'01  검출 / 파지','대상 선택과 접근'),(3.69,'02  운반 / 배치','블록을 목표 영역으로 이동'),(5.24,'03  반복','남은 블록을 다시 검출')]:
    panel(s,9.24,y,3.54,1.49,t,b,15)
note(s,'1차 미션 영상','권장 재생은 약 25초이며 원본 전체는 약 105.7초입니다. 첫 파지에서 배치까지 한 사이클을 중심으로 보여 주고 나머지는 구두로 설명합니다. 마지막 장면은 다섯 블록 배치 상태를 보여 주지만 이 파일의 재생 길이만으로 제한시간 달성을 판정하지 않습니다. 원본 영상을 삭제하거나 배속 처리하지 않았습니다.',4)

# 12 / original 13: newly confirmed stacking result with unobstructed movie.
s=slides[12]
header(s,'11  STACKING','2차 미션: 3층 적재 성공','추가 실기 결과를 반영하고, 4층 이상에서 남은 자세 제약을 구분',12)
movie(s,.55,2.14,8.42,F/'media3_054.jpg')
panel(s,9.24,2.14,3.54,1.5,'현재 성과','블록 3층 적재 성공',18)
panel(s,9.24,3.86,3.54,1.4,'남은 제약','4층 이상 접근 시\n그리퍼 자세와 도달성',16)
panel(s,9.24,5.48,3.54,1.25,'다음 검증','높이별 자세 조정과 반복 시험',15)
note(s,'3층 적재 결과','이번 PPT에 포함된 추가 영상에서 세 블록이 적층된 장면을 확인했습니다. 사용자는 3층 적재 성공 여부와 5초 유지 확인 질문에 성공했다고 답했습니다. 실험 날짜와 반복 횟수는 제공되지 않았습니다. 본문에는 3층 성공을 반영하고 기존 보고서의 미수행 상태를 최신 결과에 그대로 적용하지 않았습니다. 파일 길이는 약 54.34초이며 해제 직후의 유지 시간을 이 영상만으로 5초 이상 독립 확인했다고 주장하지 않습니다. 권장 재생 구간은 35초 부근부터 끝까지로 마지막 적재를 보여 줍니다. 4층 이상 자세 조정과 반복 시험은 남은 과제입니다.',13)

# 13 / original 12: results and bounded next steps, no wall of implementation text.
s=slides[11]
header(s,'12  CONCLUSION','확보한 성과와 다음 단계','인식, 좌표, 기구학, 센싱을 실제 장비의 미션 흐름으로 통합',13)
panel(s,.55,2.12,5.97,3.18,'확보한 성과','CV + IK 기반 5개 블록 운반 / 배치\n파지 확인과 실패 후보 재시도 구현\n카메라 관제와 캘리브레이션 도구 개발\n추가 실기에서 3층 적재 성공',18)
panel(s,6.81,2.12,5.97,3.18,'남은 검증','최종 조건의 미션 반복 성공률\n실패 후보의 시간 비용 관리\n4층 이상 적재 자세와 안정성\n사람이 확인한 결과와 센서 판정 대조',18)
rect(s,.55,5.59,12.23,1.14,'pale')
txt(s,.8,5.79,11.73,.34,'다음 일정',16,'navy',True)
txt(s,.8,6.27,11.73,.35,'최종 시연 리허설과 반복 평가 / 검증된 실행 기록을 학습 데이터로 확장',16,'navy')
note(s,'성과와 남은 검증','도구의 기능을 나열하기보다 실제로 연결한 흐름과 최신 적재 결과를 정리합니다. 반복 미션 성공률, 재시도 비용, 4층 이상 적재, 실제 결과와 센서 판정의 대조가 남아 있습니다. 9월 30일 리허설은 사용자가 제공한 원본 PPT의 팀 일정입니다. 실행 기록을 학습 데이터로 확장하는 계획은 향후 과제이며 센서 판정을 검증 없이 보상 정답으로 사용하지 않습니다. 최종보고서의 적재 상태와 이번 발표의 최신 결과는 후속으로 동기화할 필요가 있습니다.',12)

# 14 / original 14: calm closing with direct contributions available for Q&A.
s=slides[13]
line(s,.7,.69,12.63,.69,'blue',1.3)
photo(s,A/'image3.png',11.18,1.01,1.34,1.34)
txt(s,.75,1.21,9.6,.28,'TEAM SIM2REAL  /  Q&A',13,'muted')
txt(s,.75,2.16,10.4,.85,'감사합니다',34,'navy',True)
txt(s,.78,3.24,11.64,.5,'컴퓨터 비전과 역기구학 기반 블록 이동 및 적재 시스템',19,'muted')
roles=[('윤민석','비전 검출 / 미션 제어 / 자세 보정'),('김주환','학습 접근 분석 / 평가 기록 / 보고서와 발표'),('이동근','대체 파지 후보 / 관제와 캘리브레이션 도구')]
for j,(name,role) in enumerate(roles):
    y=4.41+j*.62
    txt(s,.78,y,1.25,.35,name,17,'navy',True)
    txt(s,2.3,y,10.15,.4,role,17)
line(s,.78,6.57,12.56,6.57)
txt(s,.78,6.82,11.78,.3,'부산대학교 졸업과제 세미나',12,'muted')
note(s,'질의응답','질의응답으로 전환합니다. 역할은 사용자가 확정한 최신 분담을 따릅니다. 윤민석은 비전 최적화, 자세 보정, 손목 문제와 미션 실행을, 김주환은 초기 학습 접근 분석, 정량 평가 체계, 문서와 발표를, 이동근은 대체 파지 후보, 카메라 웹 뷰어, 좌표 수집 도구와 실행 환경을 담당했습니다. 통합 실험과 결과 검토는 공동 작업입니다. 파지 확인, 캘리브레이션, MoveIt과의 차이는 별도 핵심 기술 정리 문서를 참고합니다.',14)

# Reorder the existing slide parts: embedded videos and their relationships stay attached.
ids=list(prs.slides._sldIdLst)
for el in ids: prs.slides._sldIdLst.remove(el)
for orig in ORDER: prs.slides._sldIdLst.append(ids[orig-1])
prs.core_properties.title='컴퓨터 비전과 역기구학 기반 SO-ARM101 블록 이동 및 적재 시스템'
prs.core_properties.subject='졸업과제 세미나 / 내용 및 디자인 수정본'
prs.core_properties.author='Team Sim2Real'
prs.save(DEST)
(F/'revised_text_inventory.json').write_text(json.dumps(text_inventory,ensure_ascii=False,indent=2))
(F/'slide_mapping.json').write_text(json.dumps([{'new':i+1,'original':o,'seconds':DURATIONS[i]} for i,o in enumerate(ORDER)],ensure_ascii=False,indent=2))
notes=['# 발표자 노트\n\n발표와 선택 영상 재생 합계 9분 10초, 전환 여유 50초. 영상 전체를 모두 재생하지 않는다.\n']
for i,s in enumerate(prs.slides,1): notes.append('## '+s.notes_slide.notes_text_frame.text+'\n')
(ROOT/'발표자_노트.md').write_text('\n'.join(notes))
print(DEST)
print('slides',len(prs.slides),'bytes',DEST.stat().st_size,'sha256',hashlib.sha256(DEST.read_bytes()).hexdigest())
