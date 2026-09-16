# Cloudflare Pages infrastructure

This root module creates two Cloudflare Pages Direct Upload projects, their
custom domains, DNS records, the `www` canonical redirect, and the
non-production `X-Robots-Tag` rule. `non.pllm.run` is protected by Cloudflare
Access with one-time PIN authentication for `access_email`; production remains
public.

State is AES-GCM encrypted by OpenTofu and stored in the private
`pllm-opentofu-state` R2 bucket. Infrastructure changes run separately from
GitHub Actions on a trusted workstation. The Pages deployment workflow assumes
both projects and domains already exist.

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

`tofu validate`, `plan`, and `apply` require the backend and provider credentials
plus `TF_ENCRYPTION`. Keep those operator credentials out of GitHub Actions.
