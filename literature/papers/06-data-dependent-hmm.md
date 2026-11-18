# Data-Dependent HMM with Off-Road State Determination and Real-Time Viterbi for Lane Determination

Mike Stas, Wang Hu, Jay A. Farrell. arXiv 2505.04763 (2025).

What I read on 2026-11-18: sections I–VIII. I skimmed the appendix on reference frames.

## The construction

This is an HMM over lanes, where the transition depends on the vehicle's own measured motion. The navigation system gives a position and a velocity with covariances. The predicted next position is p̂ + v̂·T, and its covariance is propagated as C_pp + 2T·C_pv + T²·C_vv + Q.

The transition probability from lane i to lane j is the probability mass of the joint Gaussian of (now, predicted next) that falls in lane i now and lane j next, divided by the probability of being in lane i now. It's computed with bivariate normal CDFs in the lane frame. An extra "not in any lane" state, L0, gets its probability as the remainder, so the chain never breaks during U-turns or in intersections. There are no tuned parameters; everything comes from the sensor covariances.

## Evidence

Two drives at 1 Hz, about 2000 points each, against RTK ground truth. 95.9% lane accuracy. The new transition alone adds about 2.25% over the earlier HMM, and transition plus emission add 5.1%. Zero breaks, where the other methods had several.

## Limits

It needs velocity and covariance from the receiver. We only have positions, so velocity would have to be estimated from the fixes. It works at lane level and assumes the road is already known.

## Where it sits in our matcher

It shows that a transition built from motion (predicted position from velocity) beats a fixed-form transition, at the same 1 Hz. The off-road state plays the same role as our skip. For us the motion has to be estimated from the track itself.
