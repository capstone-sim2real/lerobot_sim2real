"""Fully mocked browser check: no live robot/camera/server requests."""
import asyncio,json
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  b=await p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])
  page=await b.new_page(viewport={'width':1440,'height':1000});errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  commands=[];starts=[0];delay=[0]
  pixel_requests=[]
  page.on('request',lambda r:pixel_requests.append(r.url) if '/api/pixel-target?' in r.url else None)
  async def handle(route):
   url=urlsplit(route.request.url);path=url.path
   if path in ('/api/manual','/api/stop'):
    commands.append((path,route.request.post_data_json))
    return await route.fulfill(status=202,content_type='application/json',body='{}')
   if path.startswith('/api/keyboard/'):
    commands.append((path,route.request.post_data_json))
    if path.endswith('/start'):
     starts[0]+=1;sid=str(starts[0]);await asyncio.sleep(delay[0])
     return await route.fulfill(status=202,content_type='application/json',body=json.dumps({'session_id':sid}))
    return await route.fulfill(content_type='application/json',body='{}')
   if path in ['/','/app.js','/app.css','/shadcn.css','/camera-overlay.js']:
    name='index.html' if path=='/' else path[1:]
    mime='text/html' if path=='/' else 'text/css' if path.endswith('.css') else 'application/javascript'
    return await route.fulfill(content_type=mime,body=Path('src/agent/web',name).read_text())
   if path=='/overlay-renderer.js':return await route.fulfill(content_type='application/javascript',body=Path('src/camera/overlay_renderer.js').read_text())
   if path=='/api/camera/video':return await route.fulfill(content_type='image/svg+xml',body='<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#888"/></svg>')
   if path=='/api/events':return await route.fulfill(content_type='text/event-stream',body='data: {"type":"control_state","state":"idle"}\n\n')
   if path=='/api/config':data={'pixel_preview':{'camera_name':'shoulder','image_size':[1280,720],'H':[[1,0,0],[0,1,0],[0,0,1]],'base_xy_mm':[0,0],'z_mm':4.06,'radius_mm':2000,'radius_by_angle_mm':[],'angle_min_deg':-180,'angle_max_deg':180,'edge_margin_mm':0,'min_radius_mm':0,'keepout_half_width_mm':1000,'keepout_depth_mm':50},'keyboard_jog':{'heartbeat_s':.1,'speed_mm_s':20},'manual_tools':['move_arm','open_gripper'],'camera_base_url':'http://cursor.test','provider':'fake','model':'mock','places':{},'camera_view':{'stale_s':3,'poll_s':.5,'retry_s':5}}
   elif path=='/api/lease':data={'token':'test'}
   elif path=='/api/health':data={'camera_ok':True}
   elif path=='/api/pixel-target':
    q=parse_qs(url.query);data={'target':{'u':int(q['u'][0]),'v':int(q['v'][0]),'x_mm':100,'y_mm':50,'z_mm':4.06,'calibration_id':'mock'}}
   elif path=='/api/camera/config':return await route.fulfill(status=503,content_type='application/json',body='{}')
   else:data={}
   await route.fulfill(content_type='application/json',body=json.dumps(data))
  await page.route('**/*',handle)
  await page.goto('http://cursor.test/')
  await page.wait_for_function("document.querySelector('#camera').naturalWidth===1280")
  async def arm():
   await page.locator('#keyboard-toggle').click()
   assert await page.locator('#keyboard-toggle').get_attribute('aria-pressed')=='true'
  await arm()
  await page.keyboard.down('w')
  await page.wait_for_timeout(350)
  updates=[body for path,body in commands if path.endswith('/update')]
  assert len(updates)>=3,commands
  assert all(body['vector']==[1,0,0] for body in updates)
  await page.keyboard.down('a')
  await page.wait_for_timeout(140)
  assert [body for path,body in commands if path.endswith('/update')][-1]['vector']==[1,1,0]
  await page.keyboard.up('w')
  await page.wait_for_timeout(140)
  assert [body for path,body in commands if path.endswith('/update')][-1]['vector']==[0,1,0]
  await page.keyboard.up('a')
  await page.wait_for_timeout(60)
  assert commands[-1][0]=='/api/keyboard/release'
  count=len(commands);await page.wait_for_timeout(250)
  assert len(commands)==count,commands
  assert starts[0]==1
  await page.keyboard.down('j');await page.wait_for_timeout(140)
  assert [body for path,body in commands if path.endswith('/update')][-1]['vector']==[0,0,1]
  await page.evaluate("window.dispatchEvent(new Event('blur'))")
  await page.keyboard.up('j');await page.wait_for_timeout(60)
  assert commands[-1][0]=='/api/keyboard/stop'
  assert await page.locator('#keyboard-toggle').get_attribute('aria-pressed')=='false'
  # Release before slow start response: reserve and stop, never send direction.
  delay[0]=.15;await arm();count=len(commands)
  await page.keyboard.down('k');await page.keyboard.up('k')
  await page.wait_for_timeout(250)
  assert [path for path,body in commands[count:]]==['/api/keyboard/start','/api/keyboard/stop'],commands[count:]
  # One-shot actions remain one-shot; repeats and editable fields cannot jog.
  delay[0]=0
  count=len(commands)
  await page.keyboard.down('o');await page.keyboard.down('o');await page.keyboard.up('o')
  await page.wait_for_timeout(80)
  assert len(commands)==count+1 and commands[-1][0]=='/api/manual'
  await page.locator('#jog-step').focus()
  assert await page.locator('#keyboard-toggle').get_attribute('aria-pressed')=='false'
  count=len(commands);await page.keyboard.press('w');await page.wait_for_timeout(80)
  assert len(commands)==count
  await arm()
  await page.evaluate("applyControl({state:'busy',busy_with:'other'})")
  await page.keyboard.press('w');await page.wait_for_timeout(80)
  assert len(commands)==count
  await page.keyboard.press('Escape');await page.wait_for_timeout(80)
  assert commands[-1][0]=='/api/stop'
  await page.evaluate("applyControl({state:'idle'})")
  await arm();await page.keyboard.down('w');await page.wait_for_timeout(140)
  await page.keyboard.up('w');await page.wait_for_timeout(40)
  assert commands[-1][0]=='/api/keyboard/release'
  await page.evaluate("window.dispatchEvent(new Event('blur'))")
  await page.wait_for_timeout(40)
  assert commands[-1][0]=='/api/keyboard/stop'
  assert not errors,errors
  print('PASS: held heartbeat, diagonal/latest direction, release, blur, delayed start; no live requests')
  await b.close()
asyncio.run(main())
