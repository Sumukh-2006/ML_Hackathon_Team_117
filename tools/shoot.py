"""Screenshot the running app and report console errors.

Run: python tools/shoot.py [url] [out.png] [width] [height]
"""
import sys, asyncio
from playwright.async_api import async_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077/"
out = sys.argv[2] if len(sys.argv) > 2 else "notes/ui/desktop.png"
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1440
h = int(sys.argv[4]) if len(sys.argv) > 4 else 900


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        msgs = []
        pg.on("console", lambda m: msgs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: msgs.append(f"pageerror: {e}"))
        await pg.goto(url, wait_until="networkidle")
        await pg.wait_for_timeout(900)
        await pg.screenshot(path=out)
        print("\n".join(msgs) or "no console messages")
        print("saved", out)
        await b.close()


asyncio.run(main())
