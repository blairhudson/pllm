# Privacy and threat models

Learn what PLLM protects, what each party can see, and which trust assumptions still apply.

[View canonical HTML](https://pllm.run/learn/privacy-and-threat-models/)

Document ID: `pllm.docs.learn.privacy-and-threat-models`  
Release: `0.1.0`  
Build: `sha256:43a7d6d03570b59100102980d9b739320473c2b1c5a1a31e49972f24b77b95e2`  
Source hash: `sha256:b7a2162a110280a1c987c0c1e4621c5e4e3abb5f0a925f24ee1413e6cd4ae3fb`

To evaluate a private inference system, ask six questions:

1. Which values should remain private?
2. Which metadata can each party see?
3. Which parties must follow the protocol?
4. Which parties must not collude?
5. Who receives the output?
6. What evidence supports each answer?

In PLLM's prepared public-weight path, prompts, decoded output, seeds, masks,
private scales, and model state stay with the client. The preparation and inference
services both hold the public transformer body. Preparation receives stage seeds
before inference starts. Inference receives prepared corrections and masked
integer tensors later.

This path assumes that both services follow the protocol and do not collude. If
you run preparation yourself, that trust stays within your own boundary. Two
services controlled by the same operator do not provide meaningful non-collusion.

Authentication and replay controls protect the transport, but they do not protect
against every malicious participant. Timing, traffic volume, public model identity,
tensor shapes, and approximate sequence length may remain visible.

Use [research evidence records](/research/evidence/) to evaluate a specific
privacy claim. Do not infer privacy from the number or names of the services alone.
