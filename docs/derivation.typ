#set document(title: "Approach 5: derivation of the learned map matcher")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 11pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#let result(kind, name, body) = block(
  fill: luma(244), inset: 9pt, radius: 3pt, width: 100%,
)[*#kind* (#name). #body]
#let proof(body) = block(inset: (left: 10pt))[_Proof._ #body #h(1fr) $square$]
#let example(body) = block(
  stroke: (left: 1.5pt + luma(170)), inset: (left: 9pt, y: 4pt), width: 100%,
)[_Example._ #body]

#align(center)[
  #text(size: 17pt, weight: "bold")[Approach 5: derivation of the learned map matcher]
  #v(0.3em)
  Map matching on the Kubicka dataset with added noise. Written 2026-11-18 for review before any code.
]

#v(1em)

= What this document is for

The literature benchmark ended with Wöltche's MDP at 97.04% on train and 99.34% on student_test. It also showed why it is not higher on our data: its fixed 5 m thinning assumes clean GPS, and the organisers added about 5 m of noise to every fix. Approach 5 keeps the MDP's exact decoder, sets the thinning from the data, adds time and route-shape features, and learns all weights from the labelled routes.

This document derives every part of it. Only the inputs are assumed. Every claim the method rests on is proved, and every number it needs is either measured or named as a choice.

The results build on each other:
- the measure we are scored on (@measure);
- what one track becomes, and why the decoder finds the best route exactly (@lattice);
- a step-by-step error that never undercounts the measure (@error);
- the features, as penalties of simple models of driving and of GPS noise (@features);
- training, and why it optimises a bound on the measure (@learning);
- the noise model for extra training data (@noise);
- how the result will be evaluated, and what it costs to run (@evaluation).

*The running example* used throughout: a straight main road of two arcs from $A = (0, 0)$ to $B = (100, 0)$ and on to $C = (200, 0)$, each 100 m, with their reverses. A quiet road runs parallel 15 m north. Three decision points sit at $x = 20$, $x = 110$ and $x = 190$, just north of the main road. The car drove $A -> B -> C$.

= The measure we are scored on <measure>

*Inputs.* A road graph whose arcs are straight directed segments $a$ with lengths $ell(a) > 0$. Arcs joining the same two nodes in the same direction are twins and count as one road. A route is a finite sequence of arcs. For a route $P$, let $c_P (a)$ be how many times $P$ drives road $a$; this is the route read as a multiset. $L_P = sum_a c_P (a) ell(a)$ is its length. The true route is $T$.

The benchmark (`src/benchmark/metrics.py`) computes the overlap $O = sum_a min(c_P (a), c_T (a)) ell(a)$ and the correct fraction $"CF" = O \/ max(L_P, L_T)$.

#result("Definition", "route error")[
  $ E(P) = sum_a abs(c_P (a) - c_T (a)) thin ell(a). $ <eq-error>
]

#result("Lemma 1", "error is added plus missed")[
  $E(P)$ equals the added length $sum_a max(0, c_P - c_T) ell(a)$ plus the missed length $sum_a max(0, c_T - c_P) ell(a)$, and $E(P) = L_P + L_T - 2 O$.
]
#proof[For every number $x$, $abs(x) = max(0, x) + max(0, -x)$; apply it road by road. For the second identity, $abs(p - t) = p + t - 2 min(p, t)$ for any counts $p, t$; multiply by $ell(a)$ and sum.]

#result("Lemma 2", "the error controls the score")[
  $1 - "CF" <= E(P) \/ max(L_P, L_T) <= E(P) \/ L_T$.
]
#proof[$1 - "CF" = (max(L_P, L_T) - O) \/ max(L_P, L_T)$. Since $O <= min(L_P, L_T)$, $max(L_P, L_T) - O <= L_P + L_T - 2 O = E(P)$ by Lemma 1. The last step divides by the smaller of $max(L_P, L_T) >= L_T$.]

What this means for us: any method that keeps $E$ small keeps the score high. A route with $E \/ L_T <= 0.5%$ scores at least 99.5%. Everything below works on $E$, through a version of it that splits into steps.

= What a track becomes: the lattice, and exact decoding <lattice>

== Decision points

The raw fixes $z_1, ..., z_m$ have times $t_1 < ... < t_m$. Wöltche's cleaning removes duplicates and warps (a jump needing more than 1500 km/h), thins by Douglas–Peucker with tolerance $epsilon$, and merges a cloud of fixes around a standing car into its median point. The kept points $p_1, ..., p_n$ are the _decision points_. Each keeps a time: that of its fix, or the median time of its cloud.

*Measured choice of $epsilon$.* Douglas–Peucker keeps a point when it lies more than $epsilon$ from the chord through its neighbours. With noise of standard deviation $sigma$ per axis, a point on a straight road lies off the chord by about $sigma$. So with $epsilon$ near $sigma$, most points are kept: the measured cause of the MDP's failures (track 00000082: 1322 points, against 60 on its clean original).

We set $epsilon$ as the value at which the noisy train tracks keep, on average, as many points per kilometre as the clean originals keep at Wöltche's $epsilon = 5$ m. That restores the point density his method was designed for. On six tracks this lies near 20 m (00000082: 107 points; 00000010: 72 against 73). The final value is measured on all 80 train tracks and saved.

== Candidates, routes and steps

Each decision point $p_j$ gets a candidate set $K_j$: the closest positions on the 16 nearest roads within 200 m, plus the positions it adopts from its neighbours. A candidate $c$ is a position on an arc. For candidates $c in K_j$ and $c' in K_(j+1)$, the route $rho(c, c')$ is the shortest driving route from $c$ to $c'$, if one exists within the search bound.

A _step_ is a pair $(c, c')$ from neighbouring points. It is _ordinary_ when its route exists and is at most 10 times the straight distance between the points, and a _skip_ otherwise. A skip drives nothing.

Each ordinary step has a feature vector $phi_j (c, c') in RR^F$ (@features). A skip has the skip features. The cost of a step under weights $w$ is $w dot phi_j (c, c')$. A _fresh start_ at $p_j$ (at $j = 1$, or right after a skip) costs $w_"start" thin d(c)$, with $d(c)$ the candidate's distance to its point.

A _labelling_ $y = (y_1, ..., y_n)$ picks one candidate per point, $y_j in K_j$. It also records which steps are skips; a step is a skip only if it is not ordinary. Its cost is the sum of its fresh starts and its step costs:
$ C_w (y) = sum_("fresh starts" j) w_"start" thin d(y_j) + sum_(j=1)^(n-1) w dot phi_j (y_j, y_(j+1)). $ <eq-cost>

#example[At $x = 110$ the candidates are the main road eastbound and westbound at $(110, 0)$, and the parallel road at $(110, 15)$. A labelling on the main road eastbound gives three ordinary steps with short routes. One that hops to the parallel road needs a junction detour on each hop.]

== The decoder is exact

#result("Theorem 1", "exact decoding")[
  Define, backwards from the last point, $V_n (c) = 0$ for $c in K_n$ and
  $ V_j (c) = min_(c' in K_(j+1)) [ w dot phi_j (c, c') + V_(j+1)^"next" (c, c') ], $
  where $V_(j+1)^"next" (c, c') = V_(j+1) (c')$ for an ordinary step and $F_(j+1) = min_(c'') [w_"start" d(c'') + V_(j+1) (c'')]$ for a skip. Then $min_y C_w (y) = F_1$, and following the minimising choices forward from $F_1$ gives a labelling that attains it.
]
#proof[By induction from $j = n$ down: $V_j (c)$ is the least cost of finishing the labelling from point $j$ standing on $c$.
- For $j = n$ nothing is left, so the cost is 0.
- Suppose it holds for $j + 1$. Any finish from $(j, c)$ consists of one step to some $c'$, and then either an optimal finish from $(j + 1, c')$ (ordinary step) or a fresh start at $j + 1$ followed by an optimal finish (skip). The cost @eq-cost is a sum of step terms, and each term depends only on the two candidates it joins. So the best finish through a given first step is that step's cost plus the best finish after it, which is $V_(j+1) (c')$ or $F_(j+1)$ by the hypothesis. Minimising over the first step gives $V_j (c)$.

$F_1$ is the best fresh start at the first point followed by the best finish, which is the minimum of $C_w$ over all labellings.]

*Cost.* The recursion touches each step pair once: $sum_j abs(K_j) abs(K_(j+1))$ evaluations. That is between 0.06 and 1.7 million per track at the measured tolerance, against billions at 5 m.

#result("Lemma 3", "Wöltche's value iteration is this recursion")[
  On a finite layered acyclic state graph, value iteration with any discount $gamma in (0, 1]$, swept from the last layer backwards and updated in place, reaches its fixed point in one sweep. With $gamma = 1$ that fixed point is Theorem 1's.
]
#proof[A state in layer $j$ only leads to states in layer $j + 1$. A backward in-place sweep updates layer $j + 1$ before layer $j$, so each update reads final values, and a second sweep changes nothing. With $gamma = 1$ the update is exactly Theorem 1's recursion.]

So our port of the MDP and approach 5 share one decoder. We use $gamma = 1$, so the decoder minimises the same plain sum that training (@learning) works on.

= An error that splits into steps <error>

Training needs an error the decoder can add step by step (@learning). @eq-error does not split: a road's count depends on the whole route.

#example[*Why splitting naively fails.* Write $E = L_T + sum_(a in P) ell(a) (1 - 2 [a in T])$, which is exact when $P$ drives each true road at most as often as $T$ does. This looks additive. But for the zig-zag $P = (A B, B A, A B, B C)$ against $T = (A B, B C)$ it gives $200 + (-100 + 100 - 100 - 100) = 0$, while the true error is 200: the second pass over $A B$ is added road. A training method using it would be rewarded for driving true roads twice.]

== Cutting the true route into pieces

Write the chosen route as its first arc followed by step paths: $P = (a_1) ⧺ Pi_1 ⧺ dots ⧺ Pi_(n-1)$. Here $a_1$ is the arc of $y_1$, and $Pi_j$ is the arcs of step $j$'s route after the arc of $y_j$; for a skip, it is only the arc of $y_(j+1)$. The route starts at the first pick and ends at the last, so these parts are all of $P$.

Cut $T$ into pieces that are all of $T$, at any non-decreasing positions $s_1 <= ... <= s_n$ along it:
- $T_0$ is the arcs of $T$ up to and including the one containing $s_1$;
- $T_j$, for $1 <= j < n - 1$, is the arcs after the one containing $s_j$, up to and including the one containing $s_(j+1)$;
- $T_(n-1)$ is the arcs after the one containing $s_(n-1)$, to the end of $T$.

So $T = T_0 ⧺ T_1 ⧺ dots ⧺ T_(n-1)$. We take $s_j$ as $p_j$'s monotone projection onto $T$; the theorem below holds for every choice.

#result("Definition", "step error")[
  $ Delta_j (y_j, y_(j+1)) = sum_a abs(c_(Pi_j) (a) - c_(T_j) (a)) thin ell(a), quad Delta_0 (y_1) = sum_a abs([a = a_1] - c_(T_0) (a)) thin ell(a). $
]

Each $Delta_j$ depends only on the two candidates of its step, and $Delta_0$ only on the first pick. None is ever negative.

#result("Theorem 2", "the step errors never undercount")[
  For every labelling, $E(P) <= Delta_0 + sum_(j=1)^(n-1) Delta_j$.
]
#proof[Counts add over concatenation, and both cuts cover their whole route: $c_P = [a = a_1] + sum_j c_(Pi_j)$ and $c_T = c_(T_0) + sum_j c_(T_j)$. So for each road $a$,
$ abs(c_P (a) - c_T (a)) <= abs([a = a_1] - c_(T_0) (a)) + sum_j abs(c_(Pi_j) (a) - c_(T_j) (a)) $
by the triangle inequality for absolute values. Multiply by $ell(a) > 0$ and sum over roads.]

#result("Corollary", "the step errors control the score")[
  $1 - "CF" <= (Delta_0 + sum_j Delta_j) \/ L_T$.
]
#proof[Lemma 2 and Theorem 2.]

#example[The zig-zag again, with picks at $x = 20$ on $A B$, at $B$ on $B A$ (the nearest point of $B A$ to $x = 110$), and at $x = 190$ on $B C$. Its parts are $a_1 = A B$, $Pi_1 = (B A)$ and $Pi_2 = (A B, B C)$. Cutting $T$ at $x = 20, 110, 190$ gives $T_0 = (A B)$, $T_1 = (B C)$ and $T_2 = ()$. So $Delta_0 = 0$, $Delta_1 = 100 + 100$ and $Delta_2 = 100 + 100$: a total of 400 $>=$ 200 = $E$. Nothing is credited, and the zig-zag is penalised.]

The bound is tight when each step's path matches its piece. That is the situation training drives towards.

= Features: penalties of simple models <features>

A step's cost $w dot phi$ is read as a negative log-likelihood. Each feature is the penalty, up to a constant, of a simple model of one thing that holds for real driving or for GPS. A weight is that model's confidence, which is why weights are learned and not set. Every feature is divided by a scale measured on train (its median on the best reachable labellings), so the weights are comparable.

== Wöltche's geometry

- *Distance.* $d(c) + d(c')$, and extra multiples at the first and last point. A Laplace (double-exponential) model of the fix's offset from the road gives a penalty linear in $d$. A Gaussian would give $d^2$. The linear penalty grows slowly for large offsets, which suits the measured heavy tails (kurtosis 3.9 against Gaussian 3).
- *Length mismatch.* $abs(l_t - l_r) dot l_r \/ l_t$. Newson and Krumm showed $abs(l_t - l_r)$ is close to exponentially distributed on true routes; Wöltche's factor $l_r \/ l_t$ makes detours cost more than shortcuts.
- *Heading and turn.* The absolute angle between track and route headings, and between their turns. Laplace models of angular error.
- *Total turning.* The sum of absolute direction changes along the route. A Laplace prior on how much a driver turns between two points: straighter routes are more likely.

== Coverage

Let $Z_j$ be the raw fixes between $p_j$ and $p_(j+1)$. Two features: the mean over $Z_j$ of the distance from a fix to the route, and the mean over the route's points of the distance to the track through $Z_j$.

The endpoint distances look only at the two decision points. Coverage looks at the whole stretch. With $epsilon$ near 20 m, a step spans more road than before, and coverage is what tells the main road from the parallel one in between.

== Physics

*Smoothed speed.* For fix $i$, fit position against time by least squares over the $W$ fixes centred on $i$, and take the slope as velocity $hat(v)_i$.

#result("Proposition 4", "precision of the windowed speed")[
  Let fixes be 1 s apart with independent per-axis noise of variance $sigma^2$ and correlation $r(k)$ at lag $k$, and let the car move at constant velocity. Let $tau_s = s - (W - 1)\/2$ for $s = 0, ..., W - 1$. The slope estimate is unbiased, with per-axis variance
  $ "Var"(hat(v)) = sigma^2 (sum_(s, s') tau_s tau_(s') r(abs(s - s'))) \/ (sum_s tau_s^2)^2, quad r(0) = 1. $
]
#proof[The least-squares slope is $hat(v) = sum_s tau_s x_s \/ sum_s tau_s^2$, because $sum_s tau_s = 0$. With $x_s = x_0 + v tau_s + e_s$, this is $v + sum_s tau_s e_s \/ sum_s tau_s^2$, so it is unbiased. Its variance is $"Var"(sum_s tau_s e_s) \/ (sum tau_s^2)^2$, and $"Var"(sum_s tau_s e_s) = sigma^2 sum_(s, s') tau_s tau_(s') r(abs(s - s'))$.]

With our measured noise ($sigma = 5.13$ m; $r(k) = a phi^k$ with $a = 0.339$, $phi = 0.853$, @noise):

#align(center, table(
  columns: 5, align: center, stroke: 0.4pt,
  [window $W$ (fixes)], [2], [5], [11], [21],
  [error std, independent noise (m/s)], [7.25], [1.62], [0.49], [0.18],
  [error std, our correlated noise (m/s)], [6.12], [1.51], [0.57], [0.27],
))

Two things follow. One-second speeds ($W = 2$) are useless, as we had measured. And correlated noise makes long windows less effective than independent noise would suggest.

A long window also blurs real changes of speed. That bias is not in the formula, because it depends on how cars accelerate. So $W$ is measured: it is the window that makes the noisy tracks' speeds closest to their clean originals' speeds on train.

*Features.* Let $Delta t$ be the time between $p_j$ and $p_(j+1)$. Let $overline(v)$ be the mean smoothed speed over the step; across a gap, the mean of the smoothed speeds on its two sides. Let $l_r$ be the route length.

- *Speed consistency:* $abs(l_r - overline(v) Delta t)$. A Laplace model of how far the car's travel departs from what its speed implies. This is Goh et al.'s distance discrepancy, with the speed measured from positions. Across a gap it is what separates a direct route from a detour. At a standstill ($overline(v) approx 0$) it makes any movement cost.
- *Impossible speed:* $max(0, l_r \/ Delta t - v_"cap")$, with $v_"cap"$ the 99.9th percentile of true-route speeds on train.
- *Turn force:* $overline(v)^2 kappa_"max"$, where $kappa_"max"$ is the route's largest turn angle divided by the length of the shorter arc at that turn. Lateral acceleration is $v^2 kappa$, and a car cannot take a sharp corner at full speed.

== Skips

The model is a mixture: with small probability $pi$, a step is unexplainable (an off-map stretch or a large outlier). Its penalty is the constant $-log pi$, and its weight is learned. This replaces Wöltche's fixed fail cost, 3 × the track length, which a skip escaped whenever its route was over 10 times the step: the loophole measured on 00000082.

= Learning the weights <learning>

== The target, and the error relative to it

Some true routes cannot be built from the candidates. A point may have no candidate on the true road, or the driver may have taken a longer way than the shortest route. Define the _best reachable labelling_
$ y^circle.small = arg min_y Delta(y), quad Delta(y) = Delta_0 + sum_j Delta_j (y_j, y_(j+1)). $

By Theorem 1, applied with the costs $Delta_j$ in place of $w dot phi_j$, the decoder itself finds it. Its error $Delta(y^circle.small)$ is unavoidable for this lattice and is reported per track. Write $tilde(Delta)(y) = Delta(y) - Delta(y^circle.small) >= 0$.

== The training rule

Let $s_w (y) = -C_w (y)$ be a labelling's score. On one track, the structured hinge loss is
$ h(w) = max_y [ s_w (y) + tilde(Delta)(y) ] - s_w (y^circle.small). $ <eq-hinge>
The objective over training tracks $k$ is
$ J(w) = sum_k h_k (w) + lambda norm(w - w_0)^2, $ <eq-objective>
with $w_0$ Wöltche's weights for his features and 0 for the new ones, and $lambda > 0$ chosen by cross-validation.

#result("Theorem 3", "training has one best answer")[
  $J$ is convex in $w$, and with $lambda > 0$ it is strongly convex, so it has exactly one minimiser.
]
#proof[For a fixed labelling $y$, $s_w (y) + tilde(Delta)(y)$ is affine in $w$, because $C_w$ is linear in $w$ by @eq-cost. A maximum of affine functions over the finitely many labellings is convex. Subtracting the affine $s_w (y^circle.small)$ keeps it convex, and sums of convex functions are convex. $lambda norm(w - w_0)^2$ is strongly convex. A strongly convex function on $RR^F$ has exactly one minimiser.]

#result("Theorem 4", "the hinge bounds the error of the decoder's answer")[
  Let $hat(y) = arg max_y s_w (y)$ be what the decoder returns. Then $tilde(Delta)(hat(y)) <= h(w)$, and so
  $ 1 - "CF"(hat(y)) <= (Delta(y^circle.small) + h(w)) \/ L_T. $
]
#proof[Taking $y = hat(y)$ in @eq-hinge gives $h(w) >= s_w (hat(y)) + tilde(Delta)(hat(y)) - s_w (y^circle.small)$. Since $hat(y)$ maximises the score, $s_w (hat(y)) - s_w (y^circle.small) >= 0$, so $h(w) >= tilde(Delta)(hat(y))$. Then $Delta(hat(y)) = Delta(y^circle.small) + tilde(Delta)(hat(y)) <= Delta(y^circle.small) + h(w)$, and the Corollary of Theorem 2 finishes.]

This is the chain that ties training to the benchmark. Lowering the hinge lowers a bound on the score's error. The bound also has a floor: the unavoidable $Delta(y^circle.small)$, which we report.

*Computing it.* The maximiser in @eq-hinge is the decoder run with step costs $w dot phi_j - Delta_j$. This is valid because the $Delta_j$ split over steps (Theorem 2's construction), and the constant $Delta(y^circle.small)$ does not change the maximiser.

A subgradient of $J$ is $phi(y^"worst") - phi(y^circle.small) + 2 lambda (w - w_0)$, where $phi(y)$ sums the step features along $y$ and $y^"worst"$ is the maximiser of @eq-hinge. We minimise $J$ with Adam on the GPU. The features and $Delta_j$ do not depend on $w$, so they are computed once and stored.

The scale of $w$ is fixed by $w_0$ and the measured feature scales, so the weights are not free to drift together.

= Noise model for extra training data <noise>

Our tracks are the originals plus noise added by the organisers. Per axis we model the added noise as $e_t = u_t + v_t$:
- $u_t$ independent each second with variance $(1 - a) sigma^2$;
- $v_t$ a mean-reverting drift, $v_t = phi^(Delta t) v_(t - Delta t) + eta_t$, with stationary variance $a sigma^2$.

Its autocorrelation at lag $k >= 1$ s is then $a phi^k$.

Fitted to the measured autocorrelation, with $sigma = 5.13$ m:

#align(center, table(
  columns: 6, align: center, stroke: 0.4pt,
  [lag (s)], [1], [2], [3], [5], [10],
  [measured], [0.29], [0.25], [0.21], [0.15], [0.07],
  [model, $a = 0.339$, $phi = 0.853$], [0.289], [0.247], [0.211], [0.153], [0.069],
))

Both parts are Gaussian, so their sum has kurtosis 3. The measured kurtosis is 3.9, so we add rare outliers: with probability $p$ a fix gets an extra Gaussian offset of standard deviation $s$. The resulting kurtosis is $3 ((1 - p) sigma^4 + p s_"tot"^4) \/ ((1 - p) sigma^2 + p s_"tot"^2)^2$, with $s_"tot"^2 = sigma^2 + s^2$. We pick $p$ and $s$ to match the measured variance and kurtosis.

Each fitted part is checked against the measurements before any copy is made: the sample autocorrelation and kurtosis of generated noise must match the targets.

A copy is a train original plus a fresh draw of this noise, and its true route is the folder's. student_test is never used, in either version.

= Evaluation design, and cost <evaluation>

- *Folds.* The 80 train tracks are split into 5 groups. A track and all its noisy copies stay in the same group, so no copy of a validation track is ever trained on.
- *Reproduction checks.* With $w = w_0$ on Wöltche's features only, $epsilon = 5$ m and his 0.99 discount, the compiled pipeline must return byte-identical routes to our MDP on three tracks. This decides that the rebuild changes nothing before the changes begin.
- *One change at a time.* Cross-validated weighted CF, overall and per hand tag, for:
  + learned weights;
  + the measured $epsilon$;
  + coverage;
  + physics;
  + noisy copies.
- *Final test.* Train on all 80, then score student_test once. Report a paired bootstrap confidence interval (resampling tracks) for the difference from the MDP.
- *What cannot be fixed.* The per-track unavoidable error $Delta(y^circle.small)$ bounds what any weights can reach. The 14 breaks in true routes are among its causes.

*Cost.* At the measured tolerance a track has at most about 2 million step pairs, so the route searches are bounded and few. The three loops that touch every pair are compiled with Numba: the route search, the adoption test and the path walks for coverage and $Delta_j$. Everything else is array arithmetic. Each estimate is checked as soon as its stage is built:
- a few seconds per track copy;
- about 10 minutes for the 720 lattices on 16 cores;
- seconds per training epoch on the GPU.
