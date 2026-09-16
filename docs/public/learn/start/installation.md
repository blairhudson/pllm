# Installation

Install the Python package and verify the CLI without loading a model or contacting a service.

[View canonical HTML](https://pllm.run/learn/start/installation/)

Document ID: `pllm.docs.start.installation`  
Release: `0.1.0`  
Build: `sha256:88685675b040f8400eed3855030abc8a7816f2d3688fae13cf86f2739853697d`  
Source hash: `sha256:bc8da7896bdfc0868f416717d585fe986e9f8042271812477bb16b6684f1fef5`

PLLM supports Python 3.11 through 3.13. Install the command from
[PyPI](https://pypi.org/project/pllm/) with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install pllm
pllm --version
pllm --help
```

For a Python project, add the same package as a project dependency:

```bash
uv add pllm
uv run python -c "import pllm; print(pllm.__version__)"
```

`--help` and `--version` do not load a model, initialize a device, or use the
network. A successful installation confirms only that the package is available.
It does not confirm support for a particular model, protocol, deployment, or
privacy goal.

Continue to [inspect a private inference plan](/learn/start/first-private-request/).
