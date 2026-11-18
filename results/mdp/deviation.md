# Deviation from Woeltche's code: track 00000033 only

With the code's Douglas-Peucker tolerance of 5 m, our noisy track 00000033
simplifies to 1567 points. Each gets a median of 4031 candidates after
adoption, about 15 billion candidate pairs, and the run did not finish.
The clean original of the same track gives 81 points and 1 million pairs at
5 m. The organisers' added noise (std about 5 m per axis) defeats the 5 m
tolerance.

At the user's choice (2026-11-18), this one track was matched with a 20 m
tolerance: the smallest of 10, 15, 20 and 25 m that brings it under the
25.4 million pairs of track 00000021, the largest other track measured.
Every other track uses the code's defaults.
