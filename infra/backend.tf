terraform {
  backend "s3" {
    bucket                      = "pllm-opentofu-state"
    key                         = "site/terraform.tfstate"
    region                      = "auto"
    use_lockfile                = true
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }

  encryption {
    state {
      enforced = true
    }
    plan {
      enforced = true
    }
  }
}
