locals {
  zone_name          = "pllm.run"
  non_domain         = "non.pllm.run"
  production_domain  = "pllm.run"
  non_project        = "pllm-non"
  production_project = "pllm-production"
  access_otp_ids = [
    for provider in data.cloudflare_zero_trust_access_identity_providers.account.result : provider.id
    if provider.type == "onetimepin"
  ]
}

data "cloudflare_zone" "site" {
  filter = {
    account = { id = var.cloudflare_account_id }
    name    = local.zone_name
  }
}

data "cloudflare_zero_trust_access_identity_providers" "account" {
  account_id = var.cloudflare_account_id
}

check "account_has_one_time_pin" {
  assert {
    condition     = length(local.access_otp_ids) == 1
    error_message = "The Cloudflare account must expose exactly one one-time PIN Access identity provider."
  }
}

resource "cloudflare_zero_trust_access_policy" "non_owner" {
  account_id       = var.cloudflare_account_id
  decision         = "allow"
  include          = [{ email = { email = var.access_email } }]
  name             = "pllm-non-owner"
  session_duration = "24h"
}

resource "cloudflare_zero_trust_access_application" "non" {
  account_id                = var.cloudflare_account_id
  allowed_idps              = local.access_otp_ids
  app_launcher_visible      = false
  auto_redirect_to_identity = true
  domain                    = "non.pllm.run"
  name                      = "PLLM non-production"
  session_duration          = "24h"
  type                      = "self_hosted"
  policies = [
    { id = cloudflare_zero_trust_access_policy.non_owner.id, precedence = 1 },
  ]
}

resource "cloudflare_pages_project" "non" {
  account_id        = var.cloudflare_account_id
  name              = local.non_project
  production_branch = "main"
}

resource "cloudflare_pages_project" "production" {
  account_id        = var.cloudflare_account_id
  name              = local.production_project
  production_branch = "main"
}

resource "cloudflare_dns_record" "non" {
  zone_id = data.cloudflare_zone.site.id
  name    = local.non_domain
  type    = "CNAME"
  content = cloudflare_pages_project.non.subdomain
  proxied = true
  ttl     = 1
  comment = "Managed by OpenTofu for PLLM non-production Pages"
}

resource "cloudflare_dns_record" "production" {
  zone_id = data.cloudflare_zone.site.id
  name    = local.production_domain
  type    = "CNAME"
  content = cloudflare_pages_project.production.subdomain
  proxied = true
  ttl     = 1
  comment = "Managed by OpenTofu for PLLM production Pages"
}

resource "cloudflare_dns_record" "www" {
  zone_id = data.cloudflare_zone.site.id
  name    = "www.${local.production_domain}"
  type    = "CNAME"
  content = local.production_domain
  proxied = true
  ttl     = 1
  comment = "Managed by OpenTofu for PLLM canonical redirect"
}

resource "cloudflare_pages_domain" "non" {
  account_id   = var.cloudflare_account_id
  project_name = cloudflare_pages_project.non.name
  name         = local.non_domain

  depends_on = [cloudflare_dns_record.non]
}

resource "cloudflare_pages_domain" "production" {
  account_id   = var.cloudflare_account_id
  project_name = cloudflare_pages_project.production.name
  name         = local.production_domain

  depends_on = [cloudflare_dns_record.production]
}

resource "cloudflare_ruleset" "redirects" {
  zone_id     = data.cloudflare_zone.site.id
  name        = "PLLM redirects"
  description = "Canonical hostname redirects managed by OpenTofu"
  kind        = "zone"
  phase       = "http_request_dynamic_redirect"
  rules = [
    {
      action      = "redirect"
      description = "Redirect www to apex"
      enabled     = true
      expression  = "(http.host eq \"www.${local.production_domain}\")"
      ref         = "www_to_apex"
      action_parameters = {
        from_value = {
          preserve_query_string = true
          status_code           = 301
          target_url = {
            expression = "concat(\"https://${local.production_domain}\", http.request.uri.path)"
          }
        }
      }
    },
  ]

  depends_on = [cloudflare_dns_record.www]
}

resource "cloudflare_ruleset" "non_noindex" {
  zone_id     = data.cloudflare_zone.site.id
  name        = "PLLM non-production noindex"
  description = "Prevent indexing of the non-production Pages domain"
  kind        = "zone"
  phase       = "http_response_headers_transform"
  rules = [
    {
      action      = "rewrite"
      description = "Set X-Robots-Tag on non-production responses"
      enabled     = true
      expression  = "(http.host eq \"${local.non_domain}\")"
      ref         = "non_noindex"
      action_parameters = {
        headers = {
          "X-Robots-Tag" = {
            operation = "set"
            value     = "noindex, nofollow"
          }
        }
      }
    },
  ]
}
