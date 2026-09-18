# Masked linear inference

Follow seeded one-time masks through offline correction and online matrix evaluation.

[View canonical HTML](https://pllm.run/learn/masked-linear-inference/)

Document ID: `pllm.docs.learn.masked-linear-inference`  
Release: `0.1.0`  
Build: `sha256:358faf6bdfe0705a92c0f87f5cdd73fcee9a6ff7cd651f912f232815241ebaf7`  
Source hash: `sha256:e307761e359142bb900f5669aa48f929f4ca412891c019495ec5f235a93129c2`

For public matrix `W`, preparation material binds a one-time input mask `r`, output mask `s`, and ticket. Preparation computes `W*r-s` and installs it at inference. Online, the client sends `x-r`; inference combines it with the correction and returns `W*x-s`; the client adds `s`.

This changes timing, not algebra. Reusing a row reveals relations between activations, so each ticket is consumed once. The inference role must never receive the seed, and the preparation role must not retain masks or collude with inference.

Prefill packs a ticket vector and masked matrix per stage. Decode uses one ticket per stage on a persistent connection. See [masked-linear protocol](/sdk/pipeline/protocols/masked-linear/) for the component boundary.
