# Third party notices

PLLM source retains the Apache License 2.0 from the supplied runtime. The
independent lifecycle study retains its own `research/lifecycle/LICENSE`.
Dependency packages are installed separately and retain their licenses.

| Component | Role | Upstream |
| --- | --- | --- |
| Microsoft SEAL and TenSEAL | Homomorphic arithmetic | Microsoft/SEAL; OpenMined/TenSEAL |
| NumPy and PyTorch | Reference and experimental matrix kernels | numpy/numpy; pytorch/pytorch |
| FastAPI, Uvicorn, HTTPX | API and network transport | fastapi/fastapi; encode/uvicorn; encode/httpx |
| Hugging Face tools | Checkpoint and tokenizer formats | huggingface/safetensors; huggingface/tokenizers |
| Fumadocs | Documentation layout and MDX integration | fuma-nama/fumadocs |
| Next.js, React and Tailwind CSS | Documentation application | vercel/next.js; facebook/react; tailwindlabs/tailwindcss |
| OpenAI Python SDK | Optional application integration | openai/openai-python |

The documentation uses system fonts. No font files, dependency source trees or
third party model checkpoints are included. The supplied paper cites its
research sources in `paper/manuscript.md`.
