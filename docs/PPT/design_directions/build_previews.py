"""Render three slide design directions from SVG, using original report photos."""
from pathlib import Path
from html import escape
import base64
import cairo
import gi
gi.require_version('Rsvg', '2.0')
from gi.repository import Rsvg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIG = ROOT.parents[1] / 'report' / '최종보고서' / 'figures'
W, H = 1600, 900
SANS = 'NanumGothic'
SERIF = 'Noto Serif CJK KR'
MONO = 'DejaVu Sans Mono'
PHOTO = 'data:image/jpeg;base64,' + base64.b64encode((FIG / 'grasp_lifted.jpg').read_bytes()).decode()

def rect(x,y,w,h,fill,rx=0,stroke=None):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"' + (f' stroke="{stroke}" stroke-width="2"' if stroke else '') + '/>'

def text(x,y,value,size=28,color='#17191f',weight=400,family=SANS,spacing=0,anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}" letter-spacing="{spacing}" text-anchor="{anchor}">{escape(value)}</text>'

def line(x1,y1,x2,y2,color='#c6c7ce',width=2,dash=None):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>'

def photo(x,y,w,h,crop=(360,160,880,520),rx=0):
    cx,cy,cw,ch=crop
    key=f'p{x}_{y}_{w}_{h}'
    return f'<defs><clipPath id="{key}">{rect(x,y,w,h,"white",rx)}</clipPath></defs><g clip-path="url(#{key})"><svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{cx} {cy} {cw} {ch}" preserveAspectRatio="xMidYMid slice"><image href="{PHOTO}" width="1280" height="720"/></svg></g>'

def footer(color='#92959e',right='SIM2REAL / 2026'):
    return text(72,857,'윤민석 / 김주환 / 이동근',19,color)+text(1528,857,right,18,color,family=MONO,anchor='end')

def shell(content,bg):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{rect(0,0,W,H,bg)}{content}</svg>'

slides=[]

# A: large photography, restrained typography, generous white space.
a=text(78,85,'SO-ARM101 / GRADUATION PROJECT',21,'#777d88',family=MONO)
a+=text(78,298,'픽셀에서',105,weight=800)+text(78,425,'파지까지.',105,weight=800)
a+=rect(80,471,72,7,'#3163ef')
a+=text(80,545,'컴퓨터 비전과 역기구학 기반',31,'#555e6c')
a+=text(80,594,'블록 이동 및 적재 시스템',31,'#555e6c')
a+=photo(739,150,789,554,(360,165,880,520),28)
a+=text(740,743,'실제 장비 / 파지 관측 장면',20,'#7c828c')
a+=footer()
slides.append(('A_cover',shell(a,'#ffffff')))

a=text(80,72,'09  /  GRASP VERIFICATION',21,'#717987',family=MONO)
a+=text(80,162,'파지가 확인된 뒤에만 운반한다.',59,weight=800)
a+=photo(80,225,925,510,(535,270,665,415),22)
a+=text(80,778,'그리퍼 위치와 부하로 실행 결과를 확인',27,'#59616f')
for yy,label,caption,fill,fc in [(262,'PICK','접근 / 닫기','#f2f4f8','#202735'),(422,'VERIFY','위치 + 부하','#e9efff','#285adb'),(582,'TRANSPORT','파지 확인 후 운반','#f2f4f8','#202735')]:
    a+=rect(1085,yy,435,112,fill,18)
    a+=text(1118,yy+47,label,32,fc,700,MONO)
    a+=text(1118,yy+83,caption,24,'#6f7785')
for yy in [374,534]:
    a+=line(1295,yy+9,1295,yy+34,'#a5afc2',2)+text(1295,yy+46,'↓',24,'#a5afc2',anchor='middle')
a+=text(1088,766,'미확인 → 다른 후보 검토',26,'#ac6240')
a+=footer(right='DESIGN A / 09')
slides.append(('A_body',shell(a,'#ffffff')))

# B: dark background, very large type, blue state emphasis, diagram-first.
b=text(72,76,'SIM2REAL',27,'#ffffff',800,MONO,3)+text(1528,76,'SO-ARM101  /  2026',20,'#9ba8bf',family=MONO,anchor='end')
b+=line(72,111,1528,111,'#353b47',1)
b+=text(69,307,'픽셀에서',118,'#f5f7fc',900)+text(69,454,'파지까지.',118,'#5487ff',900)
b+=text(76,541,'CV + IK + FSM',30,'#c6ccd8',600,MONO,2)
b+=photo(780,172,748,430,(360,165,880,520),0)
b+=rect(780,620,748,61,'#1f4ced')
b+=text(806,659,'실제 장비에서 관찰하고, 판단하고, 실행',25,'#ffffff',700)
b+=line(73,716,1528,716,'#353b47',1)
b+=text(75,775,'컴퓨터 비전과 역기구학 기반 블록 이동 및 적재 시스템',29,'#c6ccd8')
b+=footer('#8893a7','DESIGN B / 2026')
slides.append(('B_cover',shell(b,'#0c1017')))

b=text(72,73,'09  /  GRASP VERIFICATION',22,'#8692a8',family=MONO)
b+=text(70,219,'확인되면 운반.',79,'#f5f7fc',900)
b+=text(70,330,'아니면 재시도.',79,'#5d8eff',900)
b+=text(77,429,'그리퍼 위치 + 부하',30,'#aeb8ca')
b+=text(77,483,'파지 미확인 시 운반 차단',28,'#d3d9e5')
b+=photo(838,115,690,429,(535,270,665,415),0)
b+=text(840,583,'실제 장비 / 파지 관측 장면',20,'#96a1b7')
for xx,label,desc,fill,col in [(73,'PICK','접근 / 닫기','#171d28','#edf1fa'),(584,'VERIFY','위치 + 부하 확인','#214fe9','#ffffff'),(1095,'TRANSPORT','파지 확인 후 운반','#171d28','#edf1fa')]:
    b+=rect(xx,649,433,130,fill,0)
    b+=text(xx+26,703,label,36,col,700,MONO)
    b+=text(xx+26,748,desc,25,'#e2e8f3')
b+=text(544,727,'→',37,'#5d8eff',anchor='middle')+text(1054,727,'→',37,'#5d8eff',anchor='middle')
b+=text(802,821,'미확인 → 다른 후보 검토',22,'#d6a17c',anchor='middle')
b+=text(72,866,'SO-ARM101 / SIM2REAL',18,'#707d92',family=MONO)
slides.append(('B_body',shell(b,'#0c1017')))

# C: an editorial study layout, warm paper, serif headings, precise rules.
c=text(74,68,'SIM2REAL',24,'#2e322f',700,MONO,3)+text(1526,68,'PROJECT STUDY / 2026',19,'#75776c',family=MONO,anchor='end')
c+=line(73,98,1527,98,'#95978e',1)
c+=text(72,171,'01',32,'#9c563d',family=MONO)
c+=text(72,323,'픽셀에서',106,'#262b26',600,SERIF)
c+=text(72,468,'파지까지.',106,'#262b26',600,SERIF)
c+=text(76,570,'컴퓨터 비전과 역기구학 기반',30,'#666b60')
c+=text(76,620,'블록 이동 및 적재 시스템',30,'#666b60')
c+=photo(735,162,791,551,(360,165,880,520),0)
c+=text(737,751,'FIG. 01',18,'#9c563d',family=MONO)+text(848,751,'실제 장비에서 관측한 파지 동작',20,'#676b62')
c+=line(73,805,1527,805,'#aaa99f',1)
c+=footer('#74776d','SO-ARM101 / DESIGN C')
slides.append(('C_cover',shell(c,'#f3f1e9')))

c=text(73,68,'09 / CONTACT VERIFICATION',20,'#9c563d',family=MONO)
c+=text(1527,68,'SIM2REAL / PROJECT STUDY',19,'#75776c',family=MONO,anchor='end')
c+=line(73,98,1527,98,'#95978e',1)
c+=text(73,207,'잡았다는 확인이',59,'#262b26',600,SERIF)
c+=text(73,297,'다음 동작의 조건이다.',59,'#262b26',600,SERIF)
c+=photo(764,163,763,480,(535,270,665,415),0)
c+=text(765,682,'FIG. 09',18,'#9c563d',family=MONO)+text(875,682,'실제 장비 / 파지 관측 장면',21,'#676b62')
for yy,num,title,desc in [(403,'01','PICK','접근 / 그리퍼 닫기'),(520,'02','VERIFY','그리퍼 위치와 부하 확인'),(637,'03','TRANSPORT','파지 확인 후 운반')]:
    c+=line(74,yy-21,693,yy-21,'#c3c2b6',1)
    c+=text(76,yy+28,num,25,'#9c563d',family=MONO)
    c+=text(143,yy+26,title,26,'#30362e',600,MONO)
    c+=text(143,yy+65,desc,25,'#737769')
c+=line(765,717,1527,717,'#c3c2b6',1)
c+=text(765,761,'미확인 → 다른 후보 검토',27,'#9c563d')
c+=line(73,805,1527,805,'#aaa99f',1)
c+=footer('#74776d','DESIGN C / 09')
slides.append(('C_body',shell(c,'#f3f1e9')))

pdf=cairo.PDFSurface(str(ROOT/'design_directions.pdf'),W,H)
pdf.set_metadata(cairo.PDF_METADATA_TITLE,'SO-ARM101 / Three slide design directions')
pdfctx=cairo.Context(pdf)
viewport=Rsvg.Rectangle();viewport.x=0;viewport.y=0;viewport.width=W;viewport.height=H
for name,source in slides:
    raw=source.encode()
    (ROOT/f'{name}.svg').write_bytes(raw)
    handle=Rsvg.Handle.new_from_data(raw)
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,W,H)
    handle.render_document(cairo.Context(surface),viewport)
    surface.write_to_png(str(ROOT/f'{name}.png'))
    handle.render_document(pdfctx,viewport)
    pdfctx.show_page()
pdf.finish()

font='/usr/share/fonts/naver-nanum-gothic-fonts/NanumGothicBold.ttf'
labels={'A':'A  큰 사진과 여백 / 키노트','B':'B  굵은 글자와 대비 / 테크','C':'C  차분한 활자와 정렬 / 편집 디자인'}
for variant in labels:
    board=Image.new('RGB',(1920,600),'#d9dbe0')
    draw=ImageDraw.Draw(board)
    draw.text((25,12),labels[variant],font=ImageFont.truetype(font,24),fill='#252935')
    draw.text((966,14),'본문 예시',font=ImageFont.truetype(font,20),fill='#555d69')
    for j,kind in enumerate(['cover','body']):
        im=Image.open(ROOT/f'{variant}_{kind}.png').convert('RGB').resize((944,531),Image.Resampling.LANCZOS)
        board.paste(im,(8+j*960,58))
    board.save(ROOT/f'{variant}_preview.jpg',quality=93)

print('Created 6 SVG slides, 6 PNG slides, 3 side-by-side previews and a 6-page PDF.')
