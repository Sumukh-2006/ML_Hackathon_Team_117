# Two Online Map Matching Algorithms Based on Analytic Hierarchy Process and Fuzzy Logic

Lin, Mochida, O'Neill, Yoshida, Kanazawa, Sasada. arXiv 2402.11866 (2024, a G-RIPS student project report).

What I read on 2026-11-18: sections 3–5 (related work on deep learning, the data survey, the failed learning attempts, and the AHP results). I skipped the fuzzy-logic rule tables.

## The construction

AHP: pairwise-comparison matrices over distance, direction and turn restrictions give weights for the candidate edges, and the best edge is chosen greedily, online. Fuzzy: a classic fuzzy inference system over the same cues.

## Evidence

On this dataset (which they call KCMMN) they give only qualitative figures, no numbers. They note "branches" and "jumps" near junctions.

The negative result: they tried learning-based pipelines around this dataset (a GNN with PyG, then a 1D CNN, to estimate IMU signals from BDD100K) and abandoned both. In their words, Kubicka is the only public set with ground truth, and it has too little data for deep models.

## Where it sits in our matcher

A warning, not a method: a team that tried deep learning on this dataset failed. It is evidence against an end-to-end neural matcher trained only on the 80 tracks.
