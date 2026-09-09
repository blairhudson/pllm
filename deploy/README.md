# Deployment templates

These templates build the Python package and its Rust extension with Maturin. The compiler is present only in the build stage. No Docker image has been built or deployed in this environment. The Rust extension still requires the native CI checks described in `VALIDATION.md`.

`compose.yaml` runs inference, trusted preparation, and the local gateway on one host for evaluation. In a deployment where inference is not trusted, the gateway and preparation service belong inside the customer's trust boundary. Never mount client state into the inference container.

Both matrix services require the same mounted public checkpoint. Configure distinct inference-client, preparation-client, provider-push, and local-gateway credentials. Preparation uses its fixed inference URL and provider-push credential for one-time session authorization and persistent WebSocket correction delivery; clients cannot supply callbacks or authorize inference sessions directly. All published ports bind to loopback. Use TLS and network access controls between hosts. Reverse proxies must pass WebSocket upgrades for `/v1/he/corrections/ws`, preserve `Authorization`, disable payload logging, and set an idle timeout longer than expected decode gaps. Provider rendezvous defaults to 32,768 entries, 256 MiB aggregate correction data, and a 30 second timeout; tune all three limits together.

Build the static site image from the release root with `docker build -f deploy/Dockerfile.site -t pllm-docs .`. The image's build stage requires registry access. Pin image digests and commit the resolved npm, UV and Cargo lockfiles before release.

For systemd, install the runtime at `/opt/pllm`, copy this directory to `/opt/pllm/deploy`, create the `pllm` user, resolve dependencies and build native code before enabling the unit. Systemd reads `/etc/pllm/provider.env` as root and passes its values to the service. Set the environment file to 0600 and keep checkpoints outside protected home directories.

The runtime Docker build requires `Cargo.lock` and `uv.lock` in the checkout.
Resolve and commit them first. The image reuses the locked dependency environment
and installs the built wheel without resolving dependencies again.
