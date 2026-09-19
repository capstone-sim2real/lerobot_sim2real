import asyncio,json
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  b=await p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])
  context=await b.new_context(color_scheme='dark',viewport={'width':1440,'height':1000})
  page=await context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto('http://127.0.0.1:8110',wait_until='domcontentloaded')
  await page.wait_for_function("document.querySelector('#theme-toggle').textContent==='라이트 모드'")
  assert await page.evaluate("getComputedStyle(document.body).backgroundColor==='rgb(10, 10, 10)'")
  await page.locator('.overlay-settings summary').click()
  assert await page.evaluate("getComputedStyle(document.querySelector('.overlay-options')).backgroundColor==='rgb(23, 23, 23)'")
  await page.locator('.overlay-settings summary').click()
  await page.screenshot(path='.verification/dark-desktop.png',full_page=True)
  await page.locator('#theme-toggle').click()
  await page.reload(wait_until='domcontentloaded')
  assert await page.locator('html').get_attribute('data-theme')=='light'
  await page.wait_for_function("document.querySelector('#theme-toggle').textContent==='다크 모드'")
  await page.locator('#theme-toggle').click()
  await page.emulate_media(color_scheme='light')
  assert await page.locator('html').get_attribute('data-theme')=='dark'
  await page.reload(wait_until='domcontentloaded')
  assert await page.locator('html').get_attribute('data-theme')=='dark'
  await page.set_viewport_size({'width':390,'height':844})
  assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  await page.screenshot(path='.verification/dark-mobile.png',full_page=True)
  assert not errors,errors
  print(json.dumps({'system_default':True,'manual_override':True,'reload_persistence':True,'dark_popover':True,'mobile_overflow':False,'errors':errors}))
  await b.close()
asyncio.run(main())
