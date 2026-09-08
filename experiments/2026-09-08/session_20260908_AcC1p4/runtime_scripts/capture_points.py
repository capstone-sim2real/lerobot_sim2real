"""Interactive calibration through the running teleop process; no serial access."""
import argparse,csv,json,re,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
from config import load_config
from tools.calibration_pixels import make_pixel_preview,complete_pixel_pair
p=argparse.ArgumentParser()
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--count',type=int,default=9)
p.add_argument('--wrist-limit',type=float,default=8)
p.add_argument('--snapshot-url',default='http://127.0.0.1:8090/snapshot/shoulder.jpg')
p.add_argument('--config',default='src/configs/default.yaml')
p.add_argument('--timeout',type=float,default=15)
p.add_argument('--request',type=Path,default=Path('/tmp/so101-relative-teleop/capture_request.json'))
a=p.parse_args();cfg=load_config(a.config);a.output_dir.mkdir(parents=True,exist_ok=True)
def rows():
 path=a.output_dir/'points.csv'
 return list(csv.DictReader(path.open())) if path.exists() else []
def fresh():
 with Path('/tmp/so101-relative-teleop/observations.jsonl').open('rb') as f:
  f.seek(0,2);size=f.tell();f.seek(max(0,size-16384));lines=f.read().splitlines()
 for line in reversed(lines):
  try:r=json.loads(line);break
  except ValueError:continue
 else:raise RuntimeError('No telemetry')
 if time.time()-r['time']>5:raise RuntimeError('텔레옵 기록이 오래됐습니다. 첫 터미널 상태를 확인하세요.')
 return r
try:
 for i in range(1,a.count+1):
  pattern=re.compile(rf'P{i}(?:_retry\d*)?')
  candidates=[r for r in rows() if pattern.fullmatch(r['name'])]
  if any(r.get('u_px') and r.get('v_px') for r in candidates):
   print(f'P{i}: 이미 완료, 건너뜁니다.');continue
  name=f'P{i}'
  if candidates:
   pending=candidates[-1]
   choice=input(f"{pending['name']} 팔 기록이 있습니다. 그때의 블록이 그대로면 Enter, 새로 기록하려면 r: ").strip().lower()
   if choice=='':name=pending['name'];reuse=True
   elif choice=='r':reuse=False
   else:raise RuntimeError('Enter 또는 r을 입력하세요. 다시 실행하면 이어집니다.')
  else:reuse=False
  if not reuse:
   retry=0
   while (a.output_dir/(name.lower()+'_live.json')).exists() or (a.output_dir/(name.lower()+'_live.jpg')).exists() or any(r['name']==name for r in rows()):
    retry+=1;name=f'P{i}_retry{retry}'
   input(f'[{i}/{a.count}] 블록 윗면 중앙에 기준점을 맞추고 손목 회전을 0 근처로 둔 뒤 Enter: ')
   r=fresh();roll=r['follower']['wrist_roll'];print(f'wrist_roll={roll:+.2f} deg')
   if abs(roll)>a.wrist_limit:raise RuntimeError('손목을 0 근처로 조정한 후 다시 실행하세요.')
   if a.request.exists() or a.request.with_suffix('.processing').exists():raise RuntimeError('이미 기록 요청이 처리 중입니다.')
   result=a.output_dir/(name.lower()+'_live_result.json')
   if result.exists():result.unlink() # Only a failed attempt status; images/records are preserved.
   tmp=a.request.with_suffix('.tmp');tmp.write_text(json.dumps({'name':name,'output_dir':str(a.output_dir.resolve())}));tmp.replace(a.request)
   deadline=time.monotonic()+a.timeout
   while not result.exists():
    if time.monotonic()>deadline:raise RuntimeError('기록 응답 시간 초과. 텔레옵 로그를 확인하세요. 요청을 반복하지 마세요.')
    time.sleep(.1)
   status=json.loads(result.read_text())
   if not status['ok']:raise RuntimeError(status['error'])
   subprocess.run([sys.executable,'/tmp/so101-relative-teleop/finalize_capture.py',status['record']],check=True)
  input(f'{name}: 팔 기록 완료. 집게를 벌려 블록을 그대로 두고 팔만 비킨 뒤 Enter: ')
  image=a.output_dir/(name.lower()+'_clean.jpg')
  if not image.exists():
   with urlopen(a.snapshot_url,timeout=5) as response:image.write_bytes(response.read())
  else:print('기존 깨끗한 사진을 다시 확인합니다.')
  centre,preview=make_pixel_preview(image,cfg)
  print(f'블록 중심: {centre}\n확인 이미지: {preview.resolve()}')
  import shlex
  print('로컬 PC의 다른 터미널에서 이미지 확인:')
  print('scp '+shlex.quote('ehdrms@orin-1:'+str(preview.resolve()))+' /tmp/so101-point-preview.png && xdg-open /tmp/so101-point-preview.png')
  if input('이미지의 중심 표시가 맞고 블록이 움직이지 않았으면 yes: ').strip().lower()!='yes':
   raise RuntimeError('점은 미확정으로 보존했습니다.')
  complete_pixel_pair(a.output_dir/'points.csv',name,centre,image.name)
  print(f'{name} 완료. 다음 위치로 블록을 옮기세요.')
 print(f'{a.count}점 기록 완료. 보정 적합/적용은 아직 하지 않았습니다.')
except (KeyboardInterrupt,EOFError):print('\n기록만 종료합니다. 텔레옵은 유지됩니다.')
except Exception as e:print('중단:',e);sys.exit(1)
