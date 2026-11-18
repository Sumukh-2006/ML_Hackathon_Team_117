// Dataset analysis of the map-matching corpus (Kubicka et al. 2015, as given).
// Figures: .venv/bin/python drafts/compute_stats.py && .venv/bin/python drafts/make_figures.py
#let s = json("stats/summary.json")
#let r(x, d: 1) = str(calc.round(x, digits: d))

#set document(title: "Map-matching corpus: dataset analysis")
#set page(paper: "a4", margin: (x: 2.1cm, y: 2cm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true)
#show heading: set text(font: "Noto Sans", weight: "bold")
#show heading.where(level: 1): set text(size: 11pt)
#show figure.caption: set text(size: 8.5pt)
#set figure(gap: 6pt)

#align(center)[
  #text(font: "Noto Sans", size: 15pt, weight: "bold")[Map-matching corpus: dataset analysis] \
  #v(2pt)
  #text(size: 9pt, fill: luma(90))[Kubicka et al. (IEEE IV 2015) folders as given by the organisers]
]

#v(6pt)
#table(
  columns: 6, stroke: none, align: center, inset: 4pt,
  table.hline(stroke: 0.5pt),
  [*Folders*], [*Train / test*], [*GPS fixes*], [*Driving time*], [*True routes*], [*Road arcs*],
  table.hline(stroke: 0.3pt),
  [#s.folders], [#s.train / #s.student_test], [#s.fixes], [#r(s.hours) h],
  [#r(s.route_km, d: 0) km], [#r(s.arcs_total / 1e6, d: 2) M],
  table.hline(stroke: 0.5pt),
)

Each folder is one drive. The *road graph* is the map around it: nodes (points) joined by
*arcs*, each a straight one-way piece of road. The *true route* is the answer: the arcs the car
drove, in order. The *track* is the input: one GPS fix per second, which the organisers shifted
with random noise. Folder ids #s.unseen_ids.map(str).join(", ") are not given.

= What a folder looks like
#figure(image("figures/example.svg", width: 100%),
  caption: [The task: recover the yellow route from the purple fixes.])

= Where, what, and how big
#figure(image("figures/world.svg", width: 100%),
  caption: [Location of each drive. Global, but half the folders are in Europe.])

#figure(image("figures/tags.svg", width: 62%),
  caption: [Kubicka's hand tags for difficult tracks. A folder can carry several.])

#figure(image("figures/sizes.svg", width: 100%),
  caption: [Folder sizes, log scale. Test folders fall inside the train range.])

= Time sampling
#figure(image("figures/sampling.svg", width: 100%),
  caption: [Almost every interval is 1 s. Any longer interval is what Kubicka tagged as a gap.])

= The added noise
#figure(image("figures/noise.svg", width: 100%),
  caption: [Each given fix is the original fix shifted by about #r(s.noise_std_e) m in a random
  direction; the timestamps are unchanged. Neighbouring fixes shift in similar directions.])

#figure(image("figures/noise_effect.svg", width: 100%),
  caption: [What the noise does to the input.])

= Does the track follow the true route?
#figure(image("figures/congruence.svg", width: 100%),
  caption: [Worst disagreement per folder between the original track and its true route
  (Kubicka's `mismatch_max`; our own measurement matches it within 1%). At the left end the route goes
  where the GPS did not, so the map or the annotation is wrong there. The tag marks most of that
  end, but several untagged folders sit among it.])

= Road graph and routes
#figure(image("figures/graph.svg", width: 100%),
  caption: [The map is mostly road shape points and side streets the car never used.])

#figure(image("figures/route.svg", width: 100%),
  caption: [Checking the tags against the true routes, and the #s.route_breaks_total places in
  #s.folders_with_breaks folders where the annotated route jumps between arcs that do not connect.])
