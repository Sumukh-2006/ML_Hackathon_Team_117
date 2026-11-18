"""Functional checks on the running UI: keys, scrub, layer toggles, map click.

Run: python tools/check_ui.py
"""
import asyncio
import sys
from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077/"


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        await pg.goto(BASE, wait_until="networkidle")
        await pg.wait_for_timeout(600)

        readout = lambda: pg.inner_text("#chart-readout")
        status = lambda: pg.inner_text("#tape-status")

        base = await status()
        await pg.focus("#chart")
        await pg.keyboard.press("ArrowRight")
        await pg.keyboard.press("ArrowRight")
        await pg.wait_for_timeout(150)
        after = await status()
        print("arrow key scrub:", "OK" if base != after else "FAIL", "|", after)

        await pg.keyboard.press("Shift+ArrowRight")
        await pg.wait_for_timeout(150)
        print("shift scrub:", await status())

        await pg.click("#zoom-in")
        await pg.wait_for_timeout(150)
        print("zoom in: locator visible", await pg.evaluate(
            "document.querySelector('#chart').width > 0"))

        await pg.click("#fit")
        await pg.wait_for_timeout(300)

        # layer toggle: road layer off, then on
        await pg.click('.layer[data-layer="road"]')
        pressed = await pg.get_attribute('.layer[data-layer="road"]', "aria-pressed")
        print("layer off:", pressed)
        await pg.click('.layer[data-layer="road"]')

        # click on a fix should move the scrub position
        target = await pg.evaluate("window.pathsnap.screenOf(600)")
        before = await status()
        box = await pg.locator("#chart").bounding_box()
        await pg.mouse.click(box["x"] + target[0], box["y"] + target[1])
        await pg.wait_for_timeout(200)
        now = await status()
        print("map click:", "OK" if before != now else "FAIL", "|", now)

        # keyboard miss navigation
        await pg.focus("#chart")
        await pg.keyboard.press("m")
        await pg.wait_for_timeout(700)
        print("miss toast:", await pg.inner_text("#toast"))

        # api endpoint through the UI
        await pg.click("#api-open")
        await pg.wait_for_timeout(1500)
        note = await pg.inner_text("#api-note")
        print("api:", note[:110])
        await pg.keyboard.press("Escape")

        print("page errors:", errors or "none")
        await b.close()


asyncio.run(main())
