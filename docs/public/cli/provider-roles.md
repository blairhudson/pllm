# Operate provider roles

Start authenticated inference and preparation services without exposing application APIs.

[View canonical HTML](https://pllm.run/cli/provider-roles/)

Document ID: `pllm.docs.cli.provider-roles`  
Release: `0.1.0`  
Build: `sha256:426652b6512bb11e794ef7caf6e150d2b19f0c9f7b933d041295b3bacebb441a`  
Source hash: `sha256:bfbd013d5e76b35ca98c7b602570ab9e55b90650e78453cd3c2c44e40e6be155`

Provider operators run separate computation roles:

```bash
pllm serve inference --config inference.json
pllm serve preparation --config preparation.json
```

The inference role consumes one-use prepared rows online. The trusted preparation
role computes and pushes masked corrections before a response begins. Neither role
is an application-facing Responses API or Chat Completions API endpoint.

These commands start processes only. They do not provision hosts, certificates,
identities, monitoring, rollback protection, or non-colluding operators. Keep
credentials in protected configuration or environment variables rather than
command arguments.

Read the generated references for
[`serve inference`](/cli/reference/serve/inference/) and
[`serve preparation`](/cli/reference/serve/preparation/), then apply the
[deployment boundary](/sdk/operate/deployment/) and
[preparation pipeline](/sdk/pipeline/preparation/).
