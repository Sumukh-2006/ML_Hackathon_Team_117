# Online map-matching based on Hidden Markov model for real-time traffic sensing applications

C.Y. Goh, J. Dauwels, N. Mitrovic, M.T. Asif, A. Oran, P. Jaillet. IEEE ITSC 2012.

What I read on 2026-11-18: all of it (sections I–VI). The equations were lost in text extraction, so they are reconstructed from the prose.

## The construction

An HMM whose emission is Gaussian in distance, minus a road half-width, times a speeding penalty: recorded speed against the road's speed limit. The transition comes from two physics-like measures along the A* path between candidates.

1. Distance discrepancy: the dead-reckoning distance (average of the recorded speeds × Δt) minus the path length.
2. Momentum change: velocity vectors along each segment of the path. Speed is interpolated linearly in time between the two endpoint speeds, and direction follows each segment. The measure is the average change of momentum per segment, so many abrupt turns cost more. The terminal velocity is carried into the next transition.

An SVM with an RBF kernel, trained on 3000 transitions labelled correct or incorrect, turns the two measures into a probability. σ is measured from data as 1.4826 × MAD of distances to the true route. Decoding is online Viterbi with a variable sliding window.

## Evidence

Four bus routes in Singapore, subsampled 10 s to 5 min. Path accuracy is above 0.9 below 1 min, and 0.921 optimal overall. There is no plain-HMM comparison for the transition model alone.

## Limits

It uses the speed recorded by the phone; we only have positions. The SVM is trained on single transitions, not on whole routes.

## Where it sits in our matcher

It is the closest precedent for physics features plus learned fusion inside an exact decoder. Our version measures speed from positions, and learns at the route level (structured loss) instead of per transition.
