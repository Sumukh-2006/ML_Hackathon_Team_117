# UI and screenshots

Screenshots for the technical report. Regenerate with the server running:

```bash
python tools/shoot.py http://127.0.0.1:8077/ notes/ui/desktop.png 1440 900
python tools/interactions.py      # miss view, API dialog, method dialog, student-test trip, mobile
python tools/shoot.py http://127.0.0.1:8077/ notes/ui/projector.png 1280 720
python tools/check_ui.py          # functional checks: keys, scrub, layers, map click, API
```

| file | what it shows |
|---|---|
| `desktop.png` | whole drive at fit, scores, time strip, 500 m magnifier |
| `miss.png` | a reference segment no fix reaches, locator inset |
| `trip07.png` | student-test drive, 53.7% strict vs 98.4% roads found |
| `api.png` | `POST /api/match` response with wall-clock time |
| `method.png` | scoring definitions and assumptions |
| `projector.png` | 1280×720 compact layout |
| `mobile.png` | 390 px layout |

## Design decisions

**Not a dashboard.** The subject is a map matching problem, so the page is a
chart sheet: a title block, a map, a time strip. There are no cards, no gradient
accents and no KPI tiles.

**Palette from chart making.** Buff chart paper `#EFEDE3`, ink `#14181A`,
ultramarine `#20336E` for the reference route, chart magenta `#A6124B` for the
prediction, road grey `#9AA298`. Roads are drawn with a casing stroke under a
fill stroke, the way a printed map draws them.

**Type.** Bahnschrift (a DIN descendant, installed on Windows) for every label
and number, tabular figures; Sitka for the one descriptive paragraph. No
monospace outside the API response, no all-caps eyebrows.

**One memorable element.** The time strip marks only the fixes where the snap
disagrees with the reference, so the shape of the failure is visible at a glance;
"Next miss" flies the map to a segment no fix reaches. Everything else is quiet.

**Numbers are labelled, not decorated.** Strict recall is the hero because it is
the task. The two direction rules are shown side by side, and macro F1 / log loss
are stated as N/A with a reason rather than filled with a plausible number.

## Implementation notes

- Canvas 2D, no map library and no network tiles: the demo must work offline.
- Web Mercator per trip; graticule labels, scale bar and magnifier are computed
  from the same projection as the geometry.
- Arcs shorter than a pixel are dropped at the current zoom (raster cost, no
  information).
- `prefers-reduced-motion` turns the fly-to animation into a jump.
