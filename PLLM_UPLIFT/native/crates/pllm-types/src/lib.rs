//! Compatibility contracts, NOT a security theorem or a deployed protocol registry.
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation { Public, ClientPlaintext, MaskedRing, ArithmeticLabel, BooleanLabel, AdditiveShare, Ciphertext, AttestedPlaintext }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence { Planned, ReferenceOperator, NativeOperator, Region, FullModel, IndependentReproduction }
#[derive(Clone, Debug)]
pub struct Contract {
    pub online_workers: usize,
    pub preparation_online_allowed: bool,
    pub client_model_allowed: bool,
    pub he_allowed: bool,
    pub experimental_allowed: bool,
    pub require_reviewed_composition: bool,
}
#[derive(Clone, Debug)]
pub struct Capability {
    pub id: String,
    pub input: Representation,
    pub output: Representation,
    pub online_workers: usize,
    pub needs_online_preparation: bool,
    pub needs_client_model: bool,
    pub uses_he: bool,
    pub experimental: bool,
    pub composition_reviewed: bool,
    pub installed: bool,
    pub evidence: Evidence,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic { pub code: &'static str, pub message: String }
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.code, self.message) }
}
impl std::error::Error for Diagnostic {}
fn err(code: &'static str, message: &str) -> Diagnostic { Diagnostic { code, message: message.to_owned() } }

/// Necessary admission checks. Explicit conversion operations are themselves capabilities.
pub fn validate_candidate(c: &Contract, m: &Capability) -> Result<(), Vec<Diagnostic>> {
    let mut errors = Vec::new();
    if !m.installed { errors.push(err("E_NOT_IMPLEMENTED", "method has no installed implementation")); }
    if m.online_workers != c.online_workers { errors.push(err("E_PARTIES", "online worker count differs from contract")); }
    if m.needs_online_preparation && !c.preparation_online_allowed { errors.push(err("E_ONLINE_PREPARATION", "method requires online preparation")); }
    if m.needs_client_model && !c.client_model_allowed { errors.push(err("E_CLIENT_MODEL", "method requires client weights")); }
    if m.uses_he && !c.he_allowed { errors.push(err("E_HE", "homomorphic encryption is excluded")); }
    if m.experimental && !c.experimental_allowed { errors.push(err("E_EXPERIMENTAL", "experimental method is not permitted")); }
    if c.require_reviewed_composition && !m.composition_reviewed { errors.push(err("E_ASSURANCE", "composition review is missing")); }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
pub fn validate_edge(from: &Capability, to: &Capability) -> Result<(), Diagnostic> {
    if from.output != to.input { Err(err("E_CONVERSION", "an explicit compatible conversion is required")) } else { Ok(()) }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn method() -> Capability { Capability { id: "fixture".into(), input: Representation::ArithmeticLabel, output: Representation::ArithmeticLabel, online_workers: 1, needs_online_preparation: false, needs_client_model: false, uses_he: false, experimental: false, composition_reviewed: true, installed: true, evidence: Evidence::NativeOperator } }
    fn contract() -> Contract { Contract { online_workers: 1, preparation_online_allowed: false, client_model_allowed: false, he_allowed: false, experimental_allowed: false, require_reviewed_composition: true } }
    #[test] fn admits_declared_compatible_method() { assert!(validate_candidate(&contract(), &method()).is_ok()); }
    #[test] fn rejects_worker_change() { let mut m=method(); m.online_workers=2; assert!(validate_candidate(&contract(), &m).is_err()); }
    #[test] fn rejects_missing_implementation() { let mut m=method(); m.installed=false; assert!(validate_candidate(&contract(), &m).is_err()); }
    #[test] fn no_implicit_cast() { let a=method(); let mut b=method(); b.input=Representation::BooleanLabel; assert!(validate_edge(&a,&b).is_err()); }
    #[test] fn rejects_unreviewed() { let mut m=method(); m.composition_reviewed=false; assert!(validate_candidate(&contract(), &m).is_err()); }
}
