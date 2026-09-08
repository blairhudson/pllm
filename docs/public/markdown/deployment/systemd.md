# Deploy with systemd

Run the provider as a dedicated user with a protected environment file.


## Install

Install this repository at `/opt/pllm`. Create a dedicated `pllm` system user,
resolve dependencies, and run `PLLM_NATIVE_CACHE=/var/cache/pllm/native uv run pllm build` before making the service
filesystem read only. The unit expects `/opt/pllm/.venv/bin/pllm`.

Copy `deploy/pllm-provider.service` to `/etc/systemd/system/` and configure
`/etc/pllm/provider.env` using `deploy/provider.env.example`. Make the environment
file readable only by root. Keep checkpoints outside home directories because
the unit protects those directories.

```bash
sudo chmod 600 /etc/pllm/provider.env
sudo systemctl daemon-reload
sudo systemctl enable --now pllm-provider
sudo systemctl status pllm-provider
```

The service binds to loopback. Place a configured TLS reverse proxy in front
of it for remote access, set request size and timeout limits for preparation
traffic, and disable payload logging. The local plaintext gateway remains
inside the customer's trust boundary, not on this provider host.

Discard preparation on restart unless freshness is anchored outside the
restored state. Restoring a VM snapshot is not equivalent to an ordinary crash.
