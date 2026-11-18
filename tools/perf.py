"""Load the heaviest drive and time the first paint and a scrub.

Run: python tools/perf.py
"""
import asyncio
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 900})
        msgs = []
        pg.on("pageerror", lambda e: msgs.append(str(e)))
        await pg.goto("http://127.0.0.1:8077/", wait_until="networkidle")
        await pg.wait_for_timeout(400)

        t0 = await pg.evaluate("performance.now()")
        await pg.select_option("#trip-select", "00000054")
        await pg.wait_for_function("window.pathsnap.state.trip && window.pathsnap.state.trip.id === '00000054'")
        await pg.wait_for_timeout(350)
        t1 = await pg.evaluate("performance.now()")
        print(f"load 19,313-fix drive: {t1 - t0:.0f} ms")

        # time 40 scrub steps and a redraw each
        elapsed = await pg.evaluate("""() => {
          const t = performance.now();
          for (let k = 0; k < 40; k++) window.pathsnap.setIndex(2000 + k * 200);
          return performance.now() - t;
        }""")
        print(f"40 scrub updates: {elapsed:.0f} ms ({elapsed/40:.1f} ms each)")

        frame_ms = await pg.evaluate("""async () => {
          const s = window.pathsnap.state;
          const t = performance.now();
          for (let k = 0; k < 30; k++) {
            s.view.cx += 0.00002;
            window.pathsnap.setIndex(2000 + k * 100);
            await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
          }
          return (performance.now() - t) / 30;
        }""")
        print(f"pan + redraw: {frame_ms:.1f} ms per frame")
        print("errors:", msgs or "none")
        await b.close()


asyncio.run(main())
