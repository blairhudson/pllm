use super::*;

#[test]
fn inline_failure_preserves_error_and_deepest_context() {
    let deep = AdmissionContext::new(AdmissionStage::ResourceQuote, AdmissionGate::Work);
    let outer = AdmissionContext::new(AdmissionStage::SourceCompile, AdmissionGate::ProgramConstruction);
    for error in [Error::Resource, Error::Domain, Error::Consumed, Error::Allocation] {
        let failure = AdmissionFailure::from(error);
        assert_eq!(failure.error, error);
        assert_eq!(failure.context, None);
        assert_eq!(failure.with_context_if_absent(deep).with_context_if_absent(outer).context, Some(deep));
    }
    assert_eq!((deep.node, deep.kernel, deep.required, deep.used, deep.limit), (None, None, None, None, None));
    // Inline Copy values cannot retain heap owners; no destructor or references.
    fn copy_value<T: Copy>() {}
    copy_value::<AdmissionFailure<Error>>();
    assert!(!std::mem::needs_drop::<AdmissionFailure<Error>>());
    assert!(std::mem::size_of::<AdmissionFailure<Error>>() <= 128);
}
