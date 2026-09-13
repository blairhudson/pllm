variable "cloudflare_account_id" {
  description = "Cloudflare account containing the pllm.run zone."
  type        = string
}

variable "access_email" {
  description = "Email address allowed to authenticate to non.pllm.run."
  type        = string
  default     = "blairhudson@me.com"
}
