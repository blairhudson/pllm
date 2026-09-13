output "non_pages_project" {
  value = cloudflare_pages_project.non.name
}

output "non_access_application" {
  description = "Cloudflare Access application protecting non.pllm.run."
  value       = cloudflare_zero_trust_access_application.non.id
}

output "non_site_url" {
  value = "https://${cloudflare_pages_domain.non.name}"
}

output "production_pages_project" {
  value = cloudflare_pages_project.production.name
}

output "production_site_url" {
  value = "https://${cloudflare_pages_domain.production.name}"
}
