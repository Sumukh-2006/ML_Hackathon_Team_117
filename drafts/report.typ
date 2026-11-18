// Map matching on the Kubicka corpus: the data, five models, and how they were trained.
// Part I is drafts/dataset.typ; Part II is new.
// Data:    .venv/bin/python drafts/compute_stats.py && .venv/bin/python drafts/make_figures.py
//          .venv/bin/python drafts/compute_model_stats.py && .venv/bin/python drafts/make_model_figures.py
// Build:   typst compile drafts/report.typ
#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let s = json("stats/summary.json")
#let ms = json("stats/models.json")
#let r(x, d: 1) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}
#let cf(split, model) = r(ms.cf.at(split).at(model), d: 2)

// ------------------------------------------------------------------ colours
#let ink = rgb("#0b0b0b")
#let ink2 = rgb("#52514e")
#let muted = rgb("#a3a29c")
#let grid-c = rgb("#e6e5e0")
#let c-hmm = rgb("#2a78d6")
#let c-seg = rgb("#eb6834")
#let c-mcm = rgb("#1baf7a")
#let c-mdp = rgb("#eda100")
#let c-ours = rgb("#e87ba4")
#let c-route = rgb("#f2b705")
#let c-fix = rgb("#6250d6")
#let c-road = rgb("#c9c8c2")
#let tint = rgb("#e8f1fc")

// ------------------------------------------------------------------- styles
#set document(title: "Map matching on the Kubicka corpus: data, models and training")
#set page(paper: "a4", margin: (x: 2.1cm, y: 2cm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading: set text(font: "Noto Sans", weight: "bold")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 10.5pt)
#show heading: it => block(above: 1.3em, below: 0.7em, it)
#show figure.caption: set text(size: 8.5pt)
#set figure(gap: 6pt)
#show table.cell.where(y: 0): set text(weight: "bold")
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    link(el.location(), [§#numbering(el.numbering, ..counter(heading).at(el.location()))])
  } else { it }
}

#let part(title, sub) = {
  pagebreak(weak: true)
  block(width: 100%, stroke: (bottom: 0.8pt + ink), inset: (bottom: 6pt))[
    #text(font: "Noto Sans", size: 18pt, weight: "bold", title) \
    #text(size: 10pt, fill: ink2, sub)
  ]
  v(0.6em)
}
#let tag(color, n) = box(fill: color, inset: (x: 4pt, y: 1.5pt), radius: 2pt, baseline: 1.5pt,
  text(font: "Noto Sans", weight: "bold", fill: white, size: 8pt)[#n])
#let small(body) = text(size: 7.5pt, fill: ink2, body)

// -------------------------------------------------------------------- title
#align(center)[
  #v(1.2cm)
  #text(font: "Noto Sans", size: 22pt, weight: "bold")[Map matching on the Kubicka corpus]
  #v(4pt)
  #text(font: "Noto Sans", size: 13pt, fill: ink2)[The data, five models, and how they were trained]
  #v(12pt)
  #text(size: 9.5pt, fill: ink2)[ML Hackathon · November 2026]
]
#v(0.8cm)

#block(inset: (x: 1cm))[
  *The task.* A car logs one GPS fix per second. Given those fixes and the road map around them,
  say which roads it drove, in order. This is _map matching_.

  *Part I* describes the data, including the noise the organisers added to every track.
  *Part II* explains five matchers: four from the literature, including the best published one on
  this dataset (Wöltche's MDP), and one of our own. It covers how each decides on a route, what each
  learns from the training folders, and how they score.

  *Result.* Our model keeps the MDP's exact decoder, measures its settings from the data, and learns
  its weights with an exact line search borrowed from machine translation. On the 80 noisy training
  tracks, cross-validated, it gets #r(ms.training.exact_copies.pooled, d: 2)% of the route right against
  the MDP's #cf("train", "mdp")%, mostly by avoiding two tracks where the MDP collapses. On roads no
  model was tuned on (10 held-out drives, each with 8 draws of the measured noise) the MDP is still the
  best: #cf("held_out_noisy", "mdp")% against our #cf("held_out_noisy", "learned_trained")%.
]

#v(0.6cm)
#outline(depth: 1, indent: auto)

// =========================================================================
#part("Part I · The data", [Kubicka et al. (IEEE IV 2015) folders, as given by the organisers])
// =========================================================================

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

= The added noise <added-noise>
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

// =========================================================================
#part("Part II · The models and their training", [Four matchers from the literature, and one of our own])
// =========================================================================

= How a match is scored <score>

Say the true route is 10 km and a matcher returns 10.2 km of road, of which 9.9 km is on the true
route. It added 0.3 km the car never drove and missed 0.1 km. The *correct fraction* (CF), the
score used for this dataset, divides the overlap by the longer of the two routes:
$ "CF" = "overlap" / max(L_"matched", L_"true") = 9.9 / 10.2 = 97.1%. $
Dividing by the longer route means extra road costs as much as missing road, so a matcher cannot
pad its answer. Every score below is the *weighted* CF of a set of tracks (total overlap over total
length), so long drives count more. "Error" means $100 - "CF"$.

Models are scored on three sets of tracks:
- *train*: the 80 noisy tracks with true routes. Everything that is fitted is fitted here.
- *student_test*: the 10 given test tracks, noisy, scored once per model.
- *held-out*: the 10 folder ids missing from our copy. We fetched Kubicka's clean originals for them
  after all models were frozen, then added 8 independent draws of the measured noise to each
  (@noise-model). That gives 80 noisy test tracks on roads no model has seen.

= How every matcher works <skeleton>

The five models look different in their papers, but each does the same four things
(@fig-skeleton):
+ *Thin* the track down to the points worth deciding on (_decision points_).
+ Give each point a few *candidates*: positions on nearby roads where the car might have been.
+ *Connect* each candidate to each candidate of the next point, usually by a shortest-path search.
+ *Decode*: choose one candidate per point so that a total score, added up step by step, is best.
  The chosen candidates and the paths between them are the route.

#figure(
  diagram(
    spacing: (7mm, 5mm),
    node-stroke: 0.6pt + muted, node-corner-radius: 3pt, node-inset: 6pt,
    node((0, 0), align(center)[GPS track\ #small[1 fix / s]], name: <t>, fill: rgb("#efedfb")),
    node((1, 0), align(center)[*1 Thin*\ #small[decision points]], name: <a>),
    node((2, 0), align(center)[*2 Candidates*\ #small[nearby road positions]], name: <b>),
    node((3, 0), align(center)[*3 Connect*\ #small[routes between them]], name: <c>),
    node((4, 0), align(center)[*4 Decode*\ #small[best chain]], name: <d>),
    node((5, 0), align(center)[Route\ #small[list of arcs]], name: <o>, fill: rgb("#fdf3d6")),
    edge(<t>, <a>, "-|>"), edge(<a>, <b>, "-|>"), edge(<b>, <c>, "-|>"),
    edge(<c>, <d>, "-|>"), edge(<d>, <o>, "-|>"),
  ),
  caption: [The four steps every model in this report takes.],
) <fig-skeleton>

The decoding step is easiest to see as a *lattice* (@fig-lattice). Each column holds one point's
candidates. A possible route is a path through the columns that takes one dot from each. Each link
between neighbouring columns has a cost, and the matcher wants the path with the lowest total.
There are $K^n$ such paths for $n$ points with $K$ candidates each, far too many to try. But
because the total is a sum of link costs, one sweep across the columns is enough: for every dot,
keep only the cheapest way to reach it. That takes about $n K^2$ steps and gives the exact best
path. The HMM's Viterbi algorithm and the MDP's value iteration are both this sweep; the models
differ in what a link costs.

#let lattice-fig = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let cols = (0, 2.6, 5.2, 7.8, 10.4)
  let cands = ((0.3, 1.2, 2.2), (0.5, 1.5, 2.4), (0.2, 1.1, 2.0), (0.6, 1.6), (0.4, 1.3, 2.3))
  let best = (1, 1, 0, 0, 1)
  for i in range(cols.len() - 1) {
    for a in cands.at(i) { for b in cands.at(i + 1) {
      line((cols.at(i), a), (cols.at(i + 1), b), stroke: 0.4pt + grid-c.darken(10%))
    } }
  }
  for i in range(cols.len() - 1) {
    if i != 2 {
      line((cols.at(i), cands.at(i).at(best.at(i))), (cols.at(i + 1), cands.at(i + 1).at(best.at(i + 1))),
        stroke: 1.8pt + c-ours)
    }
  }
  bezier((cols.at(2), cands.at(2).at(0)), (cols.at(3), cands.at(3).at(0)),
    (cols.at(2) + 0.9, -0.9), (cols.at(3) - 0.9, -0.9), stroke: (paint: c-ours, thickness: 1.4pt, dash: "dashed"))
  content((6.5, -0.95), text(size: 7.5pt, fill: ink2)[a skip: give up on this link, start fresh], anchor: "north")
  for i in range(cols.len()) {
    for (k, y) in cands.at(i).enumerate() {
      let on = k == best.at(i)
      circle((cols.at(i), y), radius: if on { 0.13 } else { 0.1 },
        fill: if on { c-ours } else { white }, stroke: 0.8pt + if on { c-ours } else { ink2 })
    }
    content((cols.at(i), 3.0), text(size: 8pt)[point #(i + 1)])
  }
  line((-0.6, 0.0), (-0.6, 2.6), stroke: 0.5pt + muted, mark: (end: ">", size: 0.15))
  content((-0.75, 1.3), text(size: 7.5pt, fill: ink2)[candidates], angle: 90deg, anchor: "south")
})

#figure(lattice-fig,
  caption: [A lattice of five decision points. Dots are candidate road positions, grey lines possible
  steps. The pink path is the cheapest chain. The MDP and our model may also *skip* a link that no
  sensible route explains, at a fixed price.]) <fig-lattice>

#figure(
  text(size: 8.5pt, table(
    columns: (auto, 1fr, 1fr, 1.3fr, 1fr),
    stroke: none, inset: (x: 4pt, y: 3pt), align: left,
    table.hline(stroke: 0.5pt),
    [Model], [Thin], [Candidates], [Cost of a step], [Learned from train],
    table.hline(stroke: 0.3pt),
    [#tag(c-hmm, "1") HMM], [drop fixes within $2 sigma$ of the last one kept], [one per road within 200 m],
      [distance to road + route length vs straight distance], [$sigma$ and $beta$],
    [#tag(c-seg, "2") Segment], [keep only key points (≈ 100 m apart)], [10 nearest arcs], [distance × (length + turn penalty)], [nothing],
    [#tag(c-mcm, "3") MCM], [none], [arcs within 30 m], [30 m − distance, summed], [nothing],
    [#tag(c-mdp, "4") MDP], [Douglas–Peucker, 5 m], [16 nearest + neighbours'],
      [5 weighted penalties; skip], [nothing (hand-tuned)],
    [#tag(c-ours, "5") Ours], [Douglas–Peucker, *16.6 m* (measured)], [16 nearest + neighbours'],
      [11 weighted penalties; skip], [3 settings, noise model; weights (@training)],
    table.hline(stroke: 0.5pt),
  )),
  caption: [The five models in the four steps.],
) <tab-models>

= The four models from the literature <literature>

== #tag(c-hmm, "1") Hidden Markov model #h(4pt) #text(size: 8.5pt, weight: "regular", fill: ink2)[Newson & Krumm, 2009]

The classic approach. The road position is a hidden state and each fix a noisy reading of it. A chain
of candidates is scored by two probabilities (@fig-hmm). The *emission* asks how likely the fix is if
the car was at this candidate: Gaussian in the distance $d$, so near roads win. The *transition* asks
how plausible the drive from one candidate to the next is. On real drives the distance driven between
two fixes is almost the same as the straight-line distance, so the transition is exponential in the
difference: a pair joined only by a detour is punished, however close each candidate is to its fix.

#let hmm-fig = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  line((-0.5, 0), (10.5, 0), stroke: 3pt + c-road)
  line((-0.5, 1.6), (10.5, 1.6), stroke: 3pt + c-road)
  content((10.6, 0), text(size: 7.5pt, fill: ink2)[main road], anchor: "west")
  content((10.6, 1.6), text(size: 7.5pt, fill: ink2)[parallel road], anchor: "west")
  let z1 = (2, 0.75)
  let z2 = (8, 0.55)
  line(z1, z2, stroke: (paint: c-fix, thickness: 0.8pt, dash: "dashed"))
  content((5, 0.95), text(size: 7.5pt, fill: c-fix)[straight-line distance])
  for (z, name) in ((z1, [fix $t$]), (z2, [fix $t + 1$])) {
    line(z, (z.at(0), 0), stroke: 0.6pt + ink2)
    line(z, (z.at(0), 1.6), stroke: 0.6pt + muted)
    circle((z.at(0), 0), radius: 0.09, fill: c-hmm, stroke: none)
    circle((z.at(0), 1.6), radius: 0.09, fill: white, stroke: 0.8pt + ink2)
    circle(z, radius: 0.11, fill: c-fix, stroke: none)
    if z == z1 {
      content((z.at(0) - 0.18, z.at(1) + 0.05), text(size: 7.5pt, name), anchor: "east")
    } else {
      content((z.at(0) + 0.2, z.at(1) + 0.05), text(size: 7.5pt, name), anchor: "west")
    }
  }
  content((2.15, 0.3), text(size: 7.5pt)[$d$], anchor: "west")
  line((2, -0.28), (8, -0.28), stroke: 1.4pt + c-hmm, mark: (end: ">", size: 0.18))
  content((5, -0.55), text(size: 7.5pt, fill: c-hmm)[route driven between the two candidates])
  content((1.2, 2.3), text(size: 8pt)[emission $prop e^(-d^2 \/ 2 sigma^2)$], anchor: "west")
  content((5.6, 2.3), text(size: 8pt)[transition $prop e^(-|"route" - "straight"| \/ beta)$], anchor: "west")
})
#figure(hmm-fig,
  caption: [The HMM's two questions for one step. Each fix has a candidate on both roads. The emission
  prefers the nearer road; the transition prefers pairs whose route is as long as the straight line
  between the fixes.]) <fig-hmm>

*Training.* The two widths are measured on the training routes, as the paper prescribes. $sigma$ is
the spread of fixes around the true route ($1.4826 times$ the median distance): *8.3 m*, from 208 417
fixes. It is larger than the 5.1 m of added noise because it also absorbs the original GPS error and map
offsets. $beta$ comes from the median gap between straight-line and driven distance: *0.80 m*, from
87 060 pairs. Viterbi then finds the most probable chain.

*Weak spot.* Each fix is judged on its own, so a stationary car's cloud of fixes ("hives") gets
matched to nearby side streets: train error on hives tracks is 28%.

== #tag(c-seg, "2") Segment-based feature matching #h(4pt) #text(size: 8.5pt, weight: "regular", fill: ink2)[Yin et al., 2018]

Built for sparse tracks, with the best published results at 1–5 minutes between fixes. It thins the
track to *key points* about 100 m apart and asks, between two key points: what is the cheapest drive
from a candidate arc at one to a candidate arc at the next, where cheap means staying near the
stretch of track in between, taking short roads and turning little? The search runs on an *action
graph* whose nodes are arcs and whose edges are turns (@fig-yin). Entering an arc costs its distance
from the track times (its length + 100 × a turn penalty of 0, 1 or 2).

#let yin-fig = diagram(
  spacing: (13mm, 7mm),
  node-stroke: 0.7pt + c-seg, node-corner-radius: 2pt, node-inset: 5pt,
  node((0, 1), [arc $e_1$], name: <e1>),
  node((1, 1), [arc $e_2$], name: <e2>),
  node((2, 0), [arc $e_3$], name: <e3>),
  node((2, 2), [arc $e_4$], name: <e4>),
  node((3, 1), [arc $e_5$], name: <e5>),
  edge(<e1>, <e2>, "-|>", text(size: 7pt)[straight: 0]),
  edge(<e2>, <e3>, "-|>", text(size: 7pt)[left: 1], label-side: left),
  edge(<e2>, <e4>, "-|>", text(size: 7pt)[right: 1], label-side: right),
  edge(<e3>, <e5>, "-|>", text(size: 7pt)[right: 1], label-side: left),
  edge(<e4>, <e5>, "-|>", text(size: 7pt)[left: 1], label-side: right),
  edge(<e2>, <e1>, "-|>", bend: 40deg, text(size: 7pt)[U-turn: 2], label-side: right),
)
#figure(yin-fig,
  caption: [The action graph: nodes are arcs, edges are turns, each with a turn penalty.]) <fig-yin>

*Training.* None: every value is the paper's. *Weak spot.* At 1 Hz it throws away most of the data,
and key points inside a hive land on arbitrary side streets (30% error on hives tracks).

== #tag(c-mcm, "3") Memorised multiple candidates (MCM) #h(4pt) #text(size: 8.5pt, weight: "regular", fill: ink2)[Li et al., 2023]

MCM has no probabilities. An arc within 30 m of a fix scores $30 - d$ for it. A *score matrix*
holds, for each arc and fix, the best total score of any route ending there (@fig-mcm). Each new fix
extends every route kept so far into the arcs connected to it. Routes are never dropped, so the
method never breaks. At the end it traces back from the best cell.

#let mcm-fig = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let nf = 9
  let na = 5
  let vals = (
    (22, 49, 75, 0, 0, 0, 0, 0, 0),
    (0, 0, 61, 88, 112, 0, 0, 0, 0),
    (0, 0, 0, 0, 98, 121, 147, 0, 0),
    (0, 0, 0, 74, 101, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 160, 183, 207),
  )
  let path = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (7, 4), (8, 4))
  for j in range(na) { for i in range(nf) {
    let v = vals.at(j).at(i)
    rect((i * 0.85, -j * 0.5), (i * 0.85 + 0.85, -j * 0.5 - 0.5),
      fill: if path.contains((i, j)) { c-mcm.lighten(55%) } else if v > 0 { grid-c } else { white },
      stroke: 0.4pt + muted)
    if v > 0 { content((i * 0.85 + 0.425, -j * 0.5 - 0.25), text(size: 7pt)[#v]) }
  } }
  for i in range(nf) { content((i * 0.85 + 0.425, 0.25), text(size: 7.5pt)[fix #(i + 1)]) }
  for j in range(na) { content((-0.15, -j * 0.5 - 0.25), text(size: 7.5pt)[arc #(j + 1)], anchor: "east") }
  for k in range(path.len() - 1) {
    let (i, j) = path.at(k)
    let (i2, j2) = path.at(k + 1)
    line((i * 0.85 + 0.6, -j * 0.5 - 0.25), (i2 * 0.85 + 0.25, -j2 * 0.5 - 0.25),
      stroke: 1pt + c-mcm.darken(20%), mark: (start: ">", size: 0.13))
  }
})
#figure(mcm-fig,
  caption: [MCM's score matrix (made-up numbers). Each cell is the best total score of a route ending
  on that arc at that fix. Tracing back from the best cell of the last fix gives the route (green).]) <fig-mcm>

*Training.* None; 30 m is the paper's value for this dataset. *Weak spot.* Adding up scores rewards
any road that passes near many fixes, so a side street running alongside the track earns points for
free. MCM finds the true route (recall 0.98) but adds detours to it (precision 0.80): the weakest of
the five.

== #tag(c-mdp, "4") Wöltche's MDP, the state of the art #h(4pt) #text(size: 8.5pt, weight: "regular", fill: ink2)[Wöltche, 2023] <mdp>

The best published result on this dataset: 98.73% on all 100 original tracks. It is described as a
Markov decision process solved by reinforcement learning, but nothing is trained. It is the lattice of
@fig-lattice, solved exactly by one backward sweep, with three ideas the others lack:
- *Cleaning.* Drop impossible jumps, thin by Douglas–Peucker at 5 m, and merge a stationary car's
  cloud of fixes into its median point. The merge is why it does well on hives tracks.
- *Borrowed candidates.* Each point also takes its neighbours' candidates, so an outlier fix still has
  the true road on its list.
- *Skips.* If no reasonable route joins two candidates (none, or one more than 10× the straight
  distance), the chain may skip to a fresh start for a large fixed price, so one bad fix cannot break
  the route.

A step's cost is a weighted sum of five penalties: distance to the road (weight 0.8), route length
unlike the track's (0.3), heading difference (0.5), turn difference (0.5) and total turning along the
route (0.6). Wöltche set these weights by Bayesian optimisation over about 1000 runs on hundreds of
other tracks. We ported his code and use his weights unchanged.

*Weak spot: thinning on noisy tracks.* Douglas–Peucker keeps a fix when it lies more than the
tolerance off the line through its neighbours. On a clean track, 5 m keeps only the corners. Our
tracks have about 5 m of noise on every fix, so at 5 m nearly every fix is kept (@fig-thinning). That
breaks the MDP in two ways:
- *Too many candidates.* Track 33 keeps 1567 points with about 4000 candidates each: 15 billion
  links. It never finished, so we ran that track at 20 m.
- *Too many skips.* Track 82 keeps 1322 points (its clean original: 60). With tiny steps, the
  "10× the straight distance" rule fires constantly, and the chain skips 137 times. It scores 44.5%,
  against 98.8% on the clean original. This one track is 26 of the 67 km the MDP gets wrong on train.

#figure(image("figures/models_thinning.svg", width: 100%),
  caption: [Douglas–Peucker on track 82, 600 m window. Grey: all fixes; purple: kept; yellow: true
  route. (a) At 5 m the clean original keeps a handful of points. (b) At the same 5 m the noisy track
  keeps almost all of them. (c) At our measured 16.6 m the density is back near the original's.
  Counts are for Douglas–Peucker alone; the MDP's merging removes some more.]) <fig-thinning>

= Our model <ours>

== The idea

A literature search found about fifteen neural map matchers. None reports results on this dataset:
they are trained on hundreds of thousands of sparse taxi tracks, and we have 80 dense ones. The methods
that do report here are all hand-designed, and the MDP is the best of them. So we kept the MDP's exact
decoder and changed three things:
+ *Measure the settings* from the data instead of fixing them for clean GPS.
+ *Add features* the MDP cannot see: how well a route follows the fixes _between_ two decision points,
  and whether it can be driven at the car's measured speed.
+ *Learn the weights* from the 80 annotated routes (@training).

== Measuring the settings <measured>

For every training folder we also have Kubicka's clean original track. Comparing the two isolates the
added noise, and the original is what Wöltche's settings were designed for. Three settings come out
of that comparison, using train only (@fig-measured):

- *Thinning tolerance, 16.6 m.* The tolerance at which the noisy tracks keep as many decision points as
  the clean originals keep at Wöltche's 5 m (11 435 over 80 tracks), found by bisection. This one
  number turns out to matter most.
- *Speed window, ±5 s.* Speed from one second to the next is meaningless under 5 m of noise. Fitting
  position against time over a window averages the noise out, but too long a window blurs real
  braking. ±5 s makes the noisy speed closest to the clean one: 0.29 m/s median error, against 1.08 m/s
  at ±1 s.
- *Speed cap, 60.7 m/s*: the 99.9th percentile of true speeds between decision points.

#figure(image("figures/models_measured.svg", width: 100%),
  caption: [(a) The tolerance search. (b) Speed error for each window; ±5 s is the minimum. (c) The
  noise model of @noise-model: noise regenerated from it has the measured
  autocorrelation.]) <fig-measured>

== Features

Candidates are found as in the MDP. For every pair of candidates at neighbouring points, a compiled
shortest-path search finds the route, and the step gets 11 features. Each one is a penalty that grows
as the step looks less like real driving:

#figure(
  text(size: 8.5pt, table(
    columns: (auto, 1fr, auto), stroke: none, inset: (x: 5pt, y: 3pt), align: left,
    table.hline(stroke: 0.5pt),
    [Feature], [Penalises], [From],
    table.hline(stroke: 0.3pt),
    [distance (×2)], [candidates far from their points; the last point counted separately], [MDP],
    [length], [route length unlike the straight distance; detours more than shortcuts], [MDP],
    [heading, turn], [route direction and turn unlike the track's], [MDP],
    [total turning], [winding routes], [MDP],
    [coverage (×2)], [fixes between the two points far from the route, and route far from the fixes], [new],
    [speed match], [route length unlike (measured speed × time between the points)], [new],
    [too fast], [routes that need more than 60.7 m/s], [new],
    [turn force], [sharp corners taken at speed ($v^2 times$ curvature)], [new],
    table.hline(stroke: 0.5pt),
  )),
  caption: [The 11 step features. A skip, the first point and a restart after a skip have their own
  costs, so there are 14 weights in all.],
) <tab-features>

Coverage is there because of the larger tolerance. At 16.6 m a step spans more road, and the distance
features only look at its two ends. A road running parallel 15 m away can be just as close at both
ends; only the fixes in between show which one the car was on.

The lattices are built once and stored: features, skip flags and, for training tracks, the error of
every possible step (@target). For the 90 given tracks plus 640 noisy copies that is
730 lattices with 531 million candidate pairs, built in 323 s on 16 cores. After that, decoding all 90
tracks takes 2.7 s.

== What is measured, what is set <settings>

Every number the model uses comes from one of four places:

#figure(
  text(size: 8.5pt, table(
    columns: (auto, 1fr, auto), stroke: none, inset: (x: 5pt, y: 3pt), align: left,
    table.hline(stroke: 0.5pt),
    [Source], [Settings], [How],
    table.hline(stroke: 0.3pt),
    [*Measured* on train], [thinning tolerance 16.6 m; speed window ±5 s; speed cap 60.7 m/s; the noise model (5 numbers)],
      [noisy tracks against their clean originals],
    [*Searched* on train], [9 of the 14 weights: Wöltche's six step terms, the skip price, the first-point and restart prices. The five new features stay at 0], [@training, cross-validated],
    [*Wöltche's defaults*], [16 candidates within 200 m (doubling to 10 km if empty); routes searched up to 5× the
      straight distance; skip when a route is over 10× it; clouds merged within 10 m; jumps over 1500 km/h dropped],
      [unchanged],
    [*Our choices*], [coverage from at most 16 fixes per step; 8 noisy copies per track (3 in the search); weights
      searched between 0 and 4× their start], [not tuned],
    table.hline(stroke: 0.5pt),
  )),
  caption: [Where each setting comes from. The weights start from Wöltche's values; the skip price starts
  at the median of his fixed fail cost.],
) <tab-settings>

So the operating point is estimated, the candidate machinery is Wöltche's, and the weights are what training
is for. The candidate defaults look adequate: the best chain on offer scores 99.75% (@target), so the right
road is almost always among the candidates.

*The physics.* Wöltche leaves physics out on purpose: his thinning and cloud merging change the times and
average speeds between points, so his thesis warns against speed-based terms and uses geometry only. Our speeds
avoid that problem, since they are fitted on the raw fixes and only averaged over each step. But the
three physics features (speed match, too fast, turn force) start with weight 0 and were not part of the
search we could finish (@training). So in the model as it stands, the physics is
computed and ignored. Searching all 14 weights is what would test it.

= Training <training>

Training has three parts: making enough labelled tracks, deciding what the right answer is for each,
and adjusting the weights so the decoder finds it.

== More training tracks from the noise model <noise-model>

80 drives is not much data for 14 weights, so we make more. The noise the organisers added is not
white: neighbouring fixes shift together (@added-noise). Comparing the given tracks with
their originals, we fitted a per-axis model with three parts: a fresh Gaussian offset each second, a
slow drift that keeps 85% of itself from one second to the next (together $sigma = 5.0$ m), and rare
outliers (0.6% of fixes, $sigma = 14$ m) that give the measured heavy tails. Noise drawn from the model
matches the real noise: standard deviation 5.13 m against 5.13 m, the same autocorrelation
(@fig-measured c), kurtosis 3.95 against 3.93.

A *noisy copy* is a training original plus a fresh draw of this noise. It keeps the folder's true
route, so it is a new labelled track (@fig-data). We made 8 copies of each training original. The
held-out test set (@score) is made the same way from the 10 held-out originals.

#figure(
  diagram(
    spacing: (9mm, 5mm),
    node-stroke: 0.6pt + muted, node-corner-radius: 3pt, node-inset: 5pt,
    node((0, 0), align(center)[80 clean originals], name: <o>),
    node((0, 1), align(center)[80 noisy tracks\ #small[given]], name: <n>),
    node((1, 0.5), align(center)[noise model\ #small[fitted on the difference]], name: <nm>, fill: tint),
    node((2, 0), align(center)[640 noisy copies\ #small[8 draws each]], name: <c>),
    node((2, 1), align(center)[80 held-out tracks\ #small[10 originals × 8 draws]], name: <h>),
    node((3, 0), align(center)[training], name: <l>, fill: rgb("#fbe9f0")),
    node((3, 1), align(center)[testing], name: <t>, fill: rgb("#fdf3d6")),
    edge(<o>, <nm>, "-|>"), edge(<n>, <nm>, "-|>"), edge(<nm>, <c>, "-|>"), edge(<nm>, <h>, "-|>"),
    edge(<c>, <l>, "-|>"),  edge(<h>, <t>, "-|>"),
  ),
  caption: [The noise model, fitted on train only, makes extra training tracks and the noisy held-out
  test set.],
) <fig-data>

== The right answer for each track: the oracle <target>

The true route itself is often not one of the lattice's chains: a point may have no candidate on the
true road. So the target is the *oracle*: the chain whose route is closest to the true one. To find it
with the same decoding sweep, we need an error that adds up step by step.

The score's error does not split into steps on its own, because whether a road counts as extra depends
on the whole route. The fix is to cut the true route into pieces at the decision points and charge each
step only for how its path differs from its own piece (@fig-pieces). Adding up these step errors never
gives less than the route's real error (by the triangle inequality, applied road by road), so a chain
with small step errors is guaranteed to score well.

#let pieces-fig = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  line((0, 0), (11, 0), stroke: 4pt + c-route.lighten(30%))
  content((11.2, 0), text(size: 7.5pt, fill: ink2)[true route], anchor: "west")
  let cuts = (1.0, 3.6, 6.5, 9.4)
  let cols = (c-hmm, c-seg, c-mcm)
  for i in range(3) {
    line((cuts.at(i), 0.32), (cuts.at(i + 1), 0.32), stroke: 1.5pt + cols.at(i),
      mark: (end: "|", start: "|", size: 0.15))
    content(((cuts.at(i) + cuts.at(i + 1)) / 2, 0.6), text(size: 7.5pt)[piece #(i + 1)])
  }
  for (i, x) in cuts.enumerate() {
    circle((x, 1.3), radius: 0.1, fill: c-fix, stroke: none)
    content((x, 1.6), text(size: 7.5pt)[point #(i + 1)])
    line((x, 1.2), (x, 0.05), stroke: (paint: c-fix, thickness: 0.5pt, dash: "dotted"))
  }
  line((1.0, -0.4), (3.6, -0.4), stroke: 1.5pt + c-hmm, mark: (end: ">", size: 0.15))
  bezier((3.6, -0.4), (6.5, -0.4), (4.4, -1.6), (5.7, -1.6), stroke: 1.5pt + c-seg, mark: (end: ">", size: 0.15))
  line((6.5, -0.4), (9.4, -0.4), stroke: 1.5pt + c-mcm, mark: (end: ">", size: 0.15))
  content((2.3, -0.7), text(size: 7.5pt)[error 0])
  content((5.05, -1.55), text(size: 7.5pt)[error = side road + missed piece], anchor: "north")
  content((7.95, -0.7), text(size: 7.5pt)[error 0])
})
#figure(pieces-fig,
  caption: [Step errors. The true route is cut into pieces at the decision points. Each step of a
  chain (below) is charged only for how its path differs from its own piece. Step 2 took a side road,
  so it pays for the side road and for the piece it missed.]) <fig-pieces>

The oracle chains score *99.75%* on train. That is the ceiling for these lattices, and it is well above
what any model reaches: the right chain is almost always available. Picking it is the weights' job.

== Learning the weights

We tried three methods, each fixing the previous one's problem.

*1 · Structured SVM.* The standard way to train an exact decoder: require the oracle chain to cost
less than every other chain, by a margin as large as that chain's extra error. It is convex, so it has
one best answer, and it bounds the score's error from above. Solved on the GPU over all lattices at
once, it made things slightly worse: 98.37% in-sample against 98.45% for Wöltche's weights. The bound
was too loose to steer by. It averaged about 3.4 per metre of route while the real error was 0.047, a
hundred times smaller. Most of the gap came from parallel roads: routes 11 m from the fixes but
hundreds of metres wrong, which the margin demands be separated by their full error.

*2 · Searching the real score.* This is how Wöltche set his weights (Bayesian optimisation, about 1000
runs on hundreds of tracks). The objective is now the weighted error of the routes the weights actually
produce. It changes in jumps as a weight moves, so there is no gradient. We used a coordinate search:
for each weight in turn, try it at ×0, ×¼, ×½, ×⅔, ×1.5, ×2, ×4, keep any improvement, and sweep until
nothing changes. Every search runs inside 5-fold cross-validation (search on 64 tracks, score the 16 it
never saw), first on the 64 own tracks only, then with 3 noisy copies of each added to the objective.

*3 · Exact line search (MERT).* A seven-point grid can miss the best value of a weight. Machine
translation met the same problem, tuning a few linear weights of a lattice decoder against a score that
does not split into steps, and solved it exactly: Och's _minimum error rate training_ (2003), run over
whole lattices by Macherey et al. (2008) and analysed by Dyer (2013). Move one weight, $w + t e_k$.
Every chain's cost is then a straight line in $t$, and the decoder always returns the lowest one. So the
decoded chain can only change where the *lower envelope* of those lines has a corner, and the error is
constant in between (@fig-mert). Find the corners, score one chain per piece, and the best $t$ along
that weight is found exactly, not approximately.

#let mert-fig = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let lines = ((0.4, 0.55, c-hmm), (1.6, 0.05, c-seg), (2.9, -0.35, c-mcm), (2.6, -0.05, muted))
  // top: costs
  line((0, 0), (8, 0), stroke: 0.5pt + muted, mark: (end: ">", size: 0.15))
  line((0, 0), (0, 4.2), stroke: 0.5pt + muted, mark: (end: ">", size: 0.15))
  content((8.1, 0), text(size: 7.5pt)[$t$], anchor: "west")
  content((0, 4.4), text(size: 7.5pt, fill: ink2)[cost of each chain], anchor: "south")
  for (a, b, col) in lines {
    line((0, a), (8, a + 8 * b), stroke: 0.9pt + col)
  }
  // envelope: min of the first three (the grey one is never lowest)
  let env(x) = calc.min(0.4 + 0.55 * x, 1.6 + 0.05 * x, 2.9 - 0.35 * x, 2.6 - 0.05 * x)
  let pts = range(0, 81).map(i => (i / 10, env(i / 10)))
  line(..pts, stroke: 2.4pt + c-ours.transparentize(40%))
  let x1 = 2.4
  let x2 = 3.25
  for x in (x1, x2) { line((x, -0.1), (x, 4.0), stroke: (paint: ink2, thickness: 0.5pt, dash: "dotted")) }
  content((5.4, 2.75), text(size: 7pt, fill: ink2)[grey: never lowest, never scored], anchor: "west")
  content((4.25, 0.3), text(size: 7.5pt, fill: c-ours.darken(25%))[lower envelope: what the decoder returns])
  // bottom: error steps
  let base = -2.6
  line((0, base), (8, base), stroke: 0.5pt + muted, mark: (end: ">", size: 0.15))
  line((0, base), (0, base + 1.8), stroke: 0.5pt + muted, mark: (end: ">", size: 0.15))
  content((0, base + 2.0), text(size: 7.5pt, fill: ink2)[route error], anchor: "south")
  line((0, base + 1.2), (x1, base + 1.2), stroke: 1.6pt + c-hmm)
  line((x1, base + 0.45), (x2, base + 0.45), stroke: 1.6pt + c-seg)
  line((x2, base + 0.95), (8, base + 0.95), stroke: 1.6pt + c-mcm)
  for x in (x1, x2) { line((x, base - 0.1), (x, base + 1.6), stroke: (paint: ink2, thickness: 0.5pt, dash: "dotted")) }
  content(((x1 + x2) / 2, base - 0.35), text(size: 7.5pt)[best $t$], anchor: "north")
})
#figure(mert-fig,
  caption: [The exact line search on one track. Top: as one weight moves by $t$, each chain's cost is a
  line; the decoder returns the lowest. Bottom: so the route, and its error, can only change at the
  envelope's corners. Summing these step functions over all tracks gives the total error exactly, and its
  lowest step is the best value of the weight.]) <fig-mert>

The corners are found by decoding alone. Decode at both ends of the range; if the same chain comes back,
it is the lowest on the whole interval, because the envelope is concave and lies below every line.
Otherwise decode where the two chains' lines cross, which either confirms a corner or reveals a chain in
between. On five tracks we checked it against a 41-point brute-force scan of each weight: it always found
an error at least as low, and re-scoring at its answer gave exactly the predicted error. Two things make
it fast. Along one weight every step cost is _base + t × slope_, computed once. And the route of every step
is cached, because neighbouring chains share almost all their steps. Measured on five tracks, the two
together make the exact search cost 0.6–8 s per weight, where a 41-point scan costs 6–7 s.

What the guarantee covers: each line search is exactly optimal, so every sweep can only lower the training
error, and the search stops at a point no single weight can improve. That is a guarantee about the
training tracks. It says nothing about new roads, which is what cross-validation measures.

#let tr = ms.training
#let pooled(key) = r(tr.at(key).pooled, d: 2)

*What cross-validation says* (@fig-cv). Pooled over all 80 held-out tracks, Wöltche's own weights score
#pooled("hand")%. The grid search lowers the error on the tracks it searches in every fold, and still
scores worse on the tracks it never saw: #pooled("grid_own")% on own tracks, #pooled("grid_copies")%
with noisy copies added. The exact search scores #pooled("exact_copies")%.

#figure(image("figures/models_cv.svg", width: 100%),
  caption: [Error on the held-out tracks of each fold, for each way of setting the nine searched weights
  (the new features stay at 0). Lower is better. The legend gives each method's weighted CF pooled
  over all 80 held-out tracks.]) <fig-cv>

The exact search is the only method that beats Wöltche's weights on tracks it never saw, by
#r(ms.cv_exact_minus_hand.difference, d: 2) points (paired bootstrap over the 80 tracks: 95% interval
#r(ms.cv_exact_minus_hand.low, d: 2) to #r(ms.cv_exact_minus_hand.high, d: 2), ahead in
#r(100 * ms.cv_exact_minus_hand.share_better, d: 0)% of resamples). It is better on 20 tracks, worse on
18 and unchanged on 42. That is a small, probable gain, not a proven one. Why the exact search's minima transfer
when the grid's do not, we have not established: an exact optimiser could just as well overfit harder.
We take its weights, searched once more on all 80 tracks and their copies, as the trained model.

The weights that move the same way in every fold are telling. The skip price falls from about 236 000
to between 560 and 1260: Wöltche's fail cost, set for 5 m thinning, is far too high for 16.6 m steps,
so the model was forcing implausible routes instead of skipping a bad point. The weight on the last
point's distance rises 3–10×, and the length weight 2–4×. The other weights move in different
directions in different folds, which is the part that probably does not generalise.

// =========================================================================
= Results <results>

Model 5 appears twice from here on: with Wöltche's weights, and with the trained weights of @training.

#figure(image("figures/models_overall.svg", width: 100%),
  caption: [Error ($100 - "weighted CF"$) on the three noisy test sets; lower is better. The hatched
  bar is our model with trained weights; on train its value is the cross-validated one, since the
  weights were fitted there.]) <fig-overall>

#let ci(d) = [#r(d.difference, d: 2) (#r(d.low, d: 2) to #r(d.high, d: 2))]
#let trained(split) = cf(split, "learned_trained")
#figure(
  text(size: 8.5pt, table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none, inset: (x: 4pt, y: 3pt), align: (left, right, right, right, right, right, right, right),
    table.hline(stroke: 0.5pt),
    [Weighted CF (%)], [#tag(c-hmm, "1") HMM], [#tag(c-seg, "2") Seg.], [#tag(c-mcm, "3") MCM],
      [#tag(c-mdp, "4") MDP], [#tag(c-ours, "5") \ Wöltche's w.], [#tag(c-ours, "5") \ trained],
      [trained − MDP \ (95% interval)],
    table.hline(stroke: 0.3pt),
    [train, 80 noisy], [#cf("train", "hmm")], [#cf("train", "segment")], [#cf("train", "mcm")],
      [#cf("train", "mdp")], [#cf("train", "learned")], [#r(ms.training.exact_copies.pooled, d: 2)#super[†]],
      [#ci(ms.cv_exact_minus_mdp)#super[†]],
    [student_test, 10 noisy], [#cf("student_test", "hmm")], [#cf("student_test", "segment")], [#cf("student_test", "mcm")],
      [#cf("student_test", "mdp")], [#cf("student_test", "learned")], [*#trained("student_test")*],
      [#ci(ms.trained_minus_mdp.student_test)],
    [held-out, 80 noisy], [#cf("held_out_noisy", "hmm")], [#cf("held_out_noisy", "segment")], [#cf("held_out_noisy", "mcm")],
      [*#cf("held_out_noisy", "mdp")*], [#cf("held_out_noisy", "learned")], [#trained("held_out_noisy")],
      [#ci(ms.trained_minus_mdp.held_out_noisy)],
    table.hline(stroke: 0.3pt),
    [held-out, 10 clean], [*#cf("held_out_clean", "hmm")*], [#cf("held_out_clean", "segment")], [#cf("held_out_clean", "mcm")],
      [#cf("held_out_clean", "mdp")], [#cf("held_out_clean", "learned")], [–], [],
    table.hline(stroke: 0.5pt),
  )),
  caption: [Scores on every test set. #super[†]Cross-validated: each track scored by weights searched
  without it. The interval is a paired bootstrap over tracks (10 000 resamples); on the noisy held-out
  set a track's 8 draws are resampled together. The clean row is the same 10 held-out tracks without
  added noise.],
) <tab-results>

*On roads nothing was tuned on, the MDP is still the best.* On the 80 noisy held-out tracks it leads
our trained model by #r(-ms.trained_minus_mdp.held_out_noisy.difference, d: 2) points, with the whole 95%
interval below zero. With Wöltche's weights our model trailed by
#r(-ms.ours_minus_mdp.held_out_noisy.difference, d: 2), so training closed about a third of the gap. On
the ten student_test tracks the trained model is level with the MDP: ahead by
#r(ms.trained_minus_mdp.student_test.difference, d: 2), with an interval far too wide to call.

*Training helped on both independent sets.* Against our model with Wöltche's weights, the trained
weights gain #r(ms.trained_minus_hand.student_test.difference, d: 2) points on student_test and
#r(ms.trained_minus_hand.held_out_noisy.difference, d: 2) on the noisy held-out tracks, close to the
#r(ms.cv_exact_minus_hand.difference, d: 2) cross-validation predicted. In both cases the interval just
touches zero.

*On train, ours leads, for a narrow reason.* Cross-validated, the trained model beats the MDP on train by
#r(ms.cv_exact_minus_mdp.difference, d: 1) points. @fig-pertrack shows where that comes from, using
Wöltche's weights for a like-for-like comparison: ours is better on 21 tracks and worse on 30, and the
lead comes from the two long tracks where 5 m thinning makes the MDP collapse (82 and 33, @mdp).
Thinning at 16.6 m prevents those collapses. Where 5 m does not collapse, which is most tracks and all
ten held-out ones, it keeps detail that 16.6 m throws away.

#figure(image("figures/models_pertrack.svg", width: 100%),
  caption: [Our model (Wöltche's weights) against the MDP on each train track. (a) Ours is better on 21 tracks and worse
  on 30, mostly by small amounts. (b) The tracks with the most wrongly matched road: the MDP loses
  51 km in total and ours 33 km, almost all of the difference on tracks 82 and 33.]) <fig-pertrack>

#figure(image("figures/models_tags.svg", width: 100%),
  caption: [Error on train by Kubicka's hand tag (log colour scale). Ours is best in every group except
  hives, where the MDP's thinning and cloud merging win.]) <fig-tags>

*By kind of track* (@fig-tags), ours is best on train in every group except the three *hives* tracks,
where the MDP's 3.8% error beats our 10.8%: at 16.6 m more of a parked car's cloud of fixes survives as
separate decision points.

*Noise hurts some models far more than others.* Compare each model on the same 10 held-out tracks,
clean and noisy. The HMM drops from #cf("held_out_clean", "hmm")% to #cf("held_out_noisy", "hmm")%:
its emission judges each fix alone, and 5 m of noise on every fix is enough to pull single fixes onto
nearby side roads. The MDP and our model lose 0.1–0.3 points; both look at the shape of the track,
not single fixes. Without noise the HMM is the best of all five, which is why the clean held-out test
alone would have given the wrong ranking.

*Speed.* Median time per noisy held-out track on 14 cores: HMM 0.5 s, Segment 0.3 s, MCM 0.2 s, ours
1.8 s (including building the lattice), MDP 17 s. The MDP's worst track took 6 minutes, because at
5 m it keeps nearly every fix.

= What we learned <conclusions>

- *The operating point mattered more than the model.* One measured number, the thinning tolerance,
  prevents the MDP's two collapses on train, and costs a little accuracy on every ordinary track. A
  tolerance chosen per track, 5 m where the MDP's lattice stays small and larger only where it explodes,
  should keep both. It is the next experiment.
- *The lattice is not the bottleneck.* The best chain in our lattices scores 99.75% on train, so the
  right route is nearly always on offer. Choosing it is the hard part.
- *How the weights are searched decides whether training helps.* A convex surrogate was a hundred times
  too loose to steer by, and a grid search overfitted. The exact line search from machine translation
  (MERT) was the first method to beat Wöltche's weights on unseen tracks. The gain is about 0.1–0.2
  points, consistent across cross-validation and both test sets, but not yet proven at 95%.
- *The physics features are not used yet.* They start at weight 0, and the search we finished covered only
  the nine other weights. A search over all 14 is what would test them.
- *Test under the conditions you will be scored in.* On clean tracks the plain HMM wins. With the
  organisers' noise it falls to fourth.

= References
#set par(justify: false)
#set text(size: 8.5pt)
- Dyer, C. (2013). Minimum error rate training and the convex hull semiring. arXiv:1307.3675.
- Goh, C. Y., Dauwels, J., Mitrovic, N., Asif, M. T., Oran, A., Jaillet, P. (2012). Online map-matching
  based on hidden Markov model for real-time traffic sensing applications. IEEE ITSC.
- Kubicka, M., Cela, A., Moulin, P., Mounier, H., Niculescu, S. I. (2015). Dataset for testing and
  training of map-matching algorithms. IEEE Intelligent Vehicles Symposium.
- Li, W., Wang, Y., Li, D., Xu, X. (2023). A robust map matching method by considering memorized multiple
  matching candidates. Theoretical Computer Science 941, 104–120.
- Macherey, W., Och, F. J., Thayer, I., Uszkoreit, J. (2008). Lattice-based minimum error rate training
  for statistical machine translation. EMNLP.
- Newson, P., Krumm, J. (2009). Hidden Markov map matching through noise and sparseness. ACM SIGSPATIAL GIS.
- Och, F. J. (2003). Minimum error rate training in statistical machine translation. ACL.
- Wöltche, A. (2023). Open source map matching with Markov decision processes: a new method and a detailed
  benchmark with existing approaches. Transactions in GIS 27(7), 1959–1991.
- Wöltche, A. (2024). Map matching methods for assigning vehicle positions to road networks. PhD thesis,
  University of Bamberg. doi:10.20378/irb-116612.
- Yin, Y., Shah, R. R., Wang, G., Zimmermann, R. (2018). Feature-based map matching for low-sampling-rate
  GPS trajectories. ACM Transactions on Spatial Algorithms and Systems.
