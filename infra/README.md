# Cloudflare Pages infrastructure

This root module creates two Cloudflare Pages Direct Upload projects, their
custom domains, DNS records, the `www` canonical redirect, and the
non-production `X-Robots-Tag` rule. `non.pllm.run` is protected by Cloudflare
Access with one-time PIN authentication for `access_email`; production remains
public.

State is AES-GCM encrypted by OpenTofu and stored in the private
`pllm-opentofu-state` R2 bucket. Local Restack bootstrap creates that bucket and
publishes narrow runtime credentials. `.github/workflows/provision.yml` creates
immutable encrypted plans and applies only an exact reviewed plan digest. It
also snapshots encrypted state before and after every apply.

Bootstrap locally from a trusted workstation:

```bash
export CLOUDFLARE_BOOTSTRAP_API_TOKEN=...
export GH_TOKEN=...
restack opentofu backend bootstrap
```

Neither operator credential belongs in GitHub Actions. Restack creates the R2
bucket, mints the scoped tokens declared in the repository-root `restack.toml`,
and writes tokens and the generated state passphrase directly to GitHub. Re-run
with `--mode rotate` to replace runtime tokens.

Then run the workflow from `main` in this order:

1. `plan`, confirmation `PLAN site`.
2. Review the plan, then run `apply` with its exact run ID, digest, and generated
   confirmation string.

Before the first plan, use an authenticated `cf` user profile to create the
PLLM account's Zero Trust organization and one-time PIN identity provider. This
one-time setup cannot use a generated account token because Cloudflare does not
allow account tokens to administer Access organizations or identity providers:

```bash
CLOUDFLARE_ACCOUNT_ID=... cf zero-trust organizations create \
  --auth-domain pllm.cloudflareaccess.com --name PLLM --session-duration 24h
CLOUDFLARE_ACCOUNT_ID=... cf zero-trust identity-providers create \
  --body '{"name":"One-time PIN","type":"onetimepin","config":{}}'
```

For local provider initialization and formatting without Cloudflare credentials:

```bash
tofu -chdir=infra init -backend=false
tofu -chdir=infra fmt -check -recursive
```

`tofu validate` also requires `TF_ENCRYPTION`; CI receives the complete enforced
state-and-plan configuration from the pinned Restack CLI.

Provisioning settings:

- Repository variable `CLOUDFLARE_ACCOUNT_ID`
- Repository secret `CLOUDFLARE_BACKEND_API_TOKEN`
- Repository secret `CLOUDFLARE_INFRA_API_TOKEN`
- Repository secret `PLLM_NON_PAGES_API_TOKEN`
- Repository secret `PLLM_PRODUCTION_PAGES_API_TOKEN`
- Repository secret `TOFU_STATE_PASSPHRASE`

The backend token derives one-hour, prefix-scoped credentials only for the state
bucket. The infrastructure token has account Pages and Zero Trust permissions,
plus `Zone Read`, `DNS Write`, `Dynamic URL Redirects Write`, and
`Zone Transform Rules Write` only for `pllm.run`. Separate Pages-only tokens deploy
non-production and production. The PLLM account's Access organization and
one-time PIN provider are bootstrapped through the user profile above; this
module reads the provider and owns only its PLLM application and policy.
