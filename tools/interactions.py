"""Drive the UI through its main states and save screenshots for review.

Run: python tools/interactions.py [base_url]
"""
import asyncio
import sys
from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077/"
OUT = "notes/ui"


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        msgs = []
        pg.on("console", lambda m: msgs.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        pg.on("pageerror", lambda e: msgs.append(f"pageerror: {e}"))

        await pg.goto(BASE, wait_until="networkidle")
        await pg.wait_for_timeout(500)

        await pg.click("#first-miss")
        await pg.wait_for_timeout(900)
        await pg.screenshot(path=f"{OUT}/miss.png")

        await pg.click("#api-open")
        await pg.wait_for_timeout(1200)
        await pg.screenshot(path=f"{OUT}/api.png")
        await pg.keyboard.press("Escape")

        await pg.click("#about-open")
        await pg.wait_for_timeout(600)
        await pg.screenshot(path=f"{OUT}/method.png")
        await pg.keyboard.press("Escape")

        await pg.select_option("#trip-select", "00000007")
        await pg.wait_for_timeout(1400)
        await pg.screenshot(path=f"{OUT}/trip07.png")

        await pg.set_viewport_size({"width": 390, "height": 844})
        await pg.wait_for_timeout(700)
        await pg.screenshot(path=f"{OUT}/mobile.png", full_page=True)

        print("\n".join(msgs) or "no console errors")
        await b.close()


asyncio.run(main())
