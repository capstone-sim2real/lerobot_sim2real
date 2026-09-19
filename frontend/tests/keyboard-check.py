import asyncio,json
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  b=await p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])
  page=await b.new_page(viewport={'width':1440,'height':1100})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  calls=[]
  async def mock(route):
   calls.append({'path':route.request.url.split('/')[-1],'body':route.request.post_data_json})
   await route.fulfill(status=202,content_type='application/json',body='{"accepted":true}')
  for name in ['jog','manual','stop']:await page.route('**/api/'+name,mock)
  await page.goto('http://127.0.0.1:8110',wait_until='domcontentloaded')
  await page.wait_for_function("document.querySelector('#state-badge').textContent==='대기' && !document.querySelector('[data-jog=forward]').disabled")
  await page.keyboard.press('w');assert len(calls)==0
  await page.locator('#keyboard-toggle').click()
  await page.keyboard.down('w');await page.wait_for_timeout(100)
  await page.keyboard.down('w');await page.wait_for_timeout(100)
  await page.keyboard.up('w');assert len(calls)==1,calls
  assert calls[0]['path']=='jog' and calls[0]['body']['forward_mm']==10
  for key in ['a','s','d','j','k','q','e','g','p','o','h']:
   await page.keyboard.press(key);await page.wait_for_timeout(50)
  assert len(calls)==12,calls
  await page.locator('#chat-tab').click()
  assert await page.locator('#keyboard-toggle').get_attribute('aria-pressed')=='false'
  await page.locator('#input').fill('wasd gripper');await page.keyboard.press('g')
  assert len(calls)==12
  await page.locator('#direct-tab').click();await page.locator('#keyboard-toggle').click()
  await page.evaluate("applyControl({state:'busy',busy_with:'test'})")
  await page.keyboard.press('w');assert len(calls)==12
  await page.keyboard.press('Escape');await page.wait_for_timeout(50)
  assert calls[-1]['path']=='stop'
  await page.evaluate("applyControl({state:'idle'})")
  await page.locator('#keyboard-toggle').click()
  await page.evaluate("window.dispatchEvent(new Event('blur'))")
  await page.keyboard.press('w');assert len(calls)==13
  await page.locator('#keyboard-toggle').click()
  await page.locator('#jog-step').focus()
  await page.keyboard.press('w');assert len(calls)==13
  assert await page.locator('#keyboard-toggle').get_attribute('aria-pressed')=='false'
  assert await page.locator('button[data-slot=button]').count()>20
  assert await page.evaluate("getComputedStyle(document.querySelector('#pick-here')).backgroundColor==='rgb(23, 23, 23)'")
  await page.locator('body').click(position={'x':10,'y':80})
  await page.screenshot(path='.verification/monochrome-desktop.png',full_page=True)
  await page.set_viewport_size({'width':390,'height':844})
  assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  await page.screenshot(path='.verification/monochrome-mobile.png',full_page=True)
  assert not errors,errors
  print(json.dumps({'commands':12,'repeat_blocked':True,'typing_blocked':True,'busy_blocked':True,'blur_disarms':True,'esc_stop':True,'shadcn':True,'mobile_overflow':False,'errors':errors}))
  await b.close()
asyncio.run(main())
