//! Real exported API round trips. The path module supplies public test INPUTS
//! only; all preprocessing, owners and protocol operations below use pllm_core.
#[allow(dead_code, unused_imports)]
#[path = "../src/maverick_batch/mod.rs"]
mod input_fixtures;

use input_fixtures::testing::{Fixture, FixtureCode};
use pllm_core::maverick_batch::{
    preprocess, AllocationPlan, CodeDescription, SetupInputs, UnvalidatedPublication, FIELD,
};

fn code(input: &FixtureCode) -> CodeDescription<'_> {
    CodeDescription {
        field: FIELD,
        dimension: input.dimension,
        first_permutation: &input.pi1,
        second_permutation: &input.pi2,
        first_scale: &input.v1,
        second_scale: &input.v2,
    }
}

#[test]
fn production_export_imports_and_roundtrips_both_complete_protocols() {
    let fixture = Fixture::new();
    let source = SetupInputs {
        matrices: [&fixture.matrices[0], &fixture.matrices[1]],
        privacy_codes: [code(&fixture.privacy[0]), code(&fixture.privacy[1])],
        primary_code: code(&fixture.primary),
        secondary_code: code(&fixture.secondary),
    };
    let published = preprocess(AllocationPlan::bounded().unwrap(), source).unwrap();
    let imported = UnvalidatedPublication::import(
        AllocationPlan::bounded().unwrap(),
        source,
        published.view().derived,
    )
    .unwrap();
    assert_eq!(published.view().binding, imported.view().binding);
    drop(published);
    let (mut client, mut server) = imported.validate(source).unwrap().split().unwrap();
    let (query, request) = client.issue_query(0, &fixture.inputs[0]).unwrap();
    let reply = server.evaluate(request.view()).unwrap();
    let output = client.finish_query(query, reply.view()).unwrap();
    assert_eq!(output.values(), fixture.direct(0));
    assert!(!client.has_live_window());
    let (batch, requests) = client
        .issue_batch([&fixture.inputs[0], &fixture.inputs[1]])
        .unwrap();
    let replies = [
        server.evaluate(requests[0].view()).unwrap(),
        server.evaluate(requests[1].view()).unwrap(),
    ];
    let (fixed, primary) = client
        .begin_verification(batch, &[replies[0].view(), replies[1].view()])
        .unwrap();
    let auxiliary = server.project(primary.view()).unwrap();
    let accepted = client.finish_batch(fixed, auxiliary.view()).unwrap();
    for slot in 0..2 {
        assert_eq!(accepted.outputs()[slot].values(), fixture.direct(slot));
    }
    assert!(!client.has_live_window());
    assert_eq!(client.attempts_issued(), 2);
}
