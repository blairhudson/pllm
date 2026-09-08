# Deployment templates

These templates build the Python package and its Rust extension with Maturin. The compiler is present only in the build stage. No Docker image has been built or deployed in this environment. The Rust extension still requires the native CI checks described in `VALIDATION.md`.

`compose.yaml` runs provider and local gateway on one host for evaluation. In a deployment where the provider is not trusted with plaintext, the gateway belongs on the customer's host. Never mount client keys or usable mask inventory into a provider container.

The provider requires a mounted local checkpoint and an explicit credential. The gateway requires a different credential. Both published ports bind to loopback. Use TLS and network access controls for communication between hosts.

Build the static site image from the release root with `docker build -f deploy/Dockerfile.site -t pllm-docs .`. The image's build stage requires registry access. Pin image digests and commit the resolved npm, UV and Cargo lockfiles before release.

For systemd, install the runtime at `/opt/pllm`, copy this directory to `/opt/pllm/deploy`, create the `pllm` user, resolve dependencies and build native code before enabling the unit. Systemd reads `/etc/pllm/provider.env` as root and passes its values to the service. Set the environment file to 0600 and keep checkpoints outside protected home directories.

The runtime Docker build requires `Cargo.lock` and `uv.lock` in the checkout.
Resolve and commit them first. The image reuses the locked dependency environment
and installs the built wheel without resolving dependencies again.
