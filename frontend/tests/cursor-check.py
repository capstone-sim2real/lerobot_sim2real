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
  pixel_requests=[]
  page.on('request',lambda r:pixel_requests.append(r.url) if '/api/pixel-target?' in r.url else None)
  async def handle(route):
   url=urlsplit(route.request.url);path=url.path
   if path in ['/','/app.js','/app.css','/shadcn.css','/camera-overlay.js']:
    name='index.html' if path=='/' else path[1:]
    mime='text/html' if path=='/' else 'text/css' if path.endswith('.css') else 'application/javascript'
    return await route.fulfill(content_type=mime,body=Path('src/agent/web',name).read_text())
   if path=='/overlay-renderer.js':return await route.fulfill(content_type='application/javascript',body=Path('src/camera/overlay_renderer.js').read_text())
   if path=='/api/camera/video':return await route.fulfill(content_type='image/svg+xml',body='<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#888"/></svg>')
   if path=='/api/events':return await route.fulfill(content_type='text/event-stream',body='data: {"type":"control_state","state":"idle"}\n\n')
   if path=='/api/config':data={'pixel_preview':{'camera_name':'shoulder','image_size':[1280,720],'H':[[1,0,0],[0,1,0],[0,0,1]],'base_xy_mm':[0,0],'z_mm':4.06,'radius_mm':2000,'radius_by_angle_mm':[],'angle_min_deg':-180,'angle_max_deg':180,'edge_margin_mm':0,'min_radius_mm':0,'keepout_half_width_mm':1000,'keepout_depth_mm':50},'manual_tools':[],'camera_base_url':'http://cursor.test','provider':'fake','model':'mock','places':{},'camera_view':{'stale_s':3,'poll_s':.5,'retry_s':5}}
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
  image=page.locator('#camera');rect=await image.bounding_box()
  x,y=rect['x']+100,rect['y']+100
  await page.mouse.move(x,y)
  ring=page.locator('.pixel-hover-ring')
  assert await ring.is_visible()
  assert await ring.evaluate("e=>getComputedStyle(e).opacity")=='0.35'
  assert await page.locator('#camera-wrap').evaluate("e=>getComputedStyle(e).cursor")=='none'
  await page.mouse.move(rect['x']+20,y)
  assert await ring.get_attribute('data-state')=='invalid'
  assert await ring.evaluate("e=>getComputedStyle(e).backgroundColor")=='rgb(220, 38, 38)'
  await page.mouse.move(x,y)
  assert await ring.get_attribute('data-state')=='valid'
  assert not pixel_requests,pixel_requests
  assert await page.locator('.pixel-workspace').count()==0
  await page.mouse.click(x,y)
  mark=page.locator('.pixel-selected-ring')
  await page.wait_for_function("document.querySelector('.pixel-selected-ring')?.dataset.state==='valid'")
  assert await mark.evaluate("e=>getComputedStyle(e).opacity")=='1'
  original=[await mark.get_attribute(k) for k in ('cx','cy')]
  await page.mouse.move(x+80,y+60)
  assert original==[await mark.get_attribute(k) for k in ('cx','cy')]
  await page.mouse.move(5,5)
  assert not await ring.is_visible() and await mark.is_visible()
  await page.set_viewport_size({'width':390,'height':844})
  await page.wait_for_timeout(100)
  bounds=await mark.bounding_box()
  image_bounds=await image.bounding_box()
  assert abs(bounds['width']-21*image_bounds['width']/1280)<1,bounds
  assert await mark.get_attribute('r')=='9'
  assert await mark.evaluate("e=>getComputedStyle(e).fill")== 'rgb(37, 99, 235)'
  assert original==[await mark.get_attribute(k) for k in ('cx','cy')]
  assert len(pixel_requests)==1,pixel_requests
  assert not errors,errors
  print('PASS: translucent hover, hidden native cursor, solid persistent selection, pointer leave, original small blue marker, local hover without requests, no JS errors')
  await b.close()
asyncio.run(main())
