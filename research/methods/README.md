# Method portfolio

`registry.json` is the canonical registry of reproduction-target aliases. Publication records live
in `sources/`, method identities in `records/`, artifact locks in `source-locks/`, and explanatory
cards in `cards/`. `references.bib` preserves citations.

Registry status is literal: cards are not installed methods, source-reported claims are not PLLM
measurements, and null source hashes/commits remain acquisition gates. `fulltext_gate: true` means
available metadata was insufficient for faithful reproduction; no gated paper is reproduced here.
Acquisition status and workflow status are independent; acquired source does not imply a build,
execution, fidelity result, or reproduction.

Recipes live in [`../recipes/`](../recipes/) and assurance specifications in
[`../assurance/`](../assurance/). Run `python3 research/validate.py` from repository root after
changing any registry, card, source lock, or recipe.
