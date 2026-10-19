use super::*;

#[test]
fn exporter_contract_shapes_without_long_replay() {
    let (f, t) = frozen();
    assert_eq!(selected(&f).len(), 66);
    for case in selected(&f) {
        validate_case(case).unwrap();
    }
    assert_eq!(gate_header().as_object().unwrap().len(), 8);
    assert_eq!(table_header().as_object().unwrap().len(), 9);
    assert_eq!(t["case_count"], 18560);
    assert_eq!(
        t["tables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["count"].as_u64().unwrap())
            .sum::<u64>(),
        18560
    );
    assert_eq!(
        t["tables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["count"].as_u64().unwrap().div_ceil(4))
            .sum::<u64>(),
        4640
    );
    assert!(implementation().len() <= 256);
    assert_eq!(lookup::CHUNK_ENTRIES, 256);
}

#[test]
fn malformed_producer_cases_reject_before_issuance() {
    assert!(validate_opt_in(None, true).is_err());
    assert!(validate_opt_in(Some("0"), true).is_err());
    assert!(validate_opt_in(Some("1"), false).is_err());
    validate_opt_in(Some("1"), true).unwrap();
    for case in [
        json!({}),
        json!({"id":"missing-args","gate":"compare"}),
        json!({"id":"bad","gate":"unimplemented","args":[]}),
        json!({"id":"bad","gate":"shift","args":[0,0,0,"TR"]}),
        json!({"id":"bad","gate":"shift","args":[0,8,8,"TR"]}),
        json!({"id":"bad","gate":"shift","args":[256,8,6,"ARS"]}),
        json!({"id":"bad","gate":"b2a","args":[2,0,50]}),
        json!({"id":"bad","gate":"compare","args":[128,8,0,"signed"]}),
        json!({"id":"bad","gate":"compare","args":[0,8,0,"low",9]}),
        json!({"id":"bad","gate":"compare","args":[0,8,257,"unsigned",null]}),
        json!({"id":"bad","gate":"phi-select","args":[1,1u64<<50,0]}),
        json!({"id":"bad","gate":"external","args":[-1,0]}),
    ] {
        assert!(validate_case(&case).is_err(), "accepted {case}");
    }
    let values = [3, 7];
    let meta = json!({"name":"silu","input_bits":50,"output_bits":50,"start":0,
        "stop":2,"count":2,"sha256_u64le":words_hash(&values)});
    validate_table(&meta, "silu", &values).unwrap();
    for field in [
        "name",
        "input_bits",
        "output_bits",
        "start",
        "stop",
        "count",
        "sha256_u64le",
    ] {
        let mut missing = meta.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(validate_table(&missing, "silu", &values).is_err());
    }
    assert!(validate_table(&meta, "silu", &[3, 8]).is_err());
    assert!(validate_table(&meta, "silu", &[]).is_err());
}

#[test]
fn tiny_fixture_computes_outputs_and_drains_actual_owners() {
    for (case, expected) in [
        (
            json!({"id":"tiny-tr","gate":"shift","args":[191,8,6,"TR"]}),
            json!({"bits":2,"word":2}),
        ),
        (
            json!({"id":"tiny-ars","gate":"shift","args":[255,8,6,"ARS"]}),
            json!({"bits":8,"word":255}),
        ),
        (
            json!({"id":"tiny-b2a","gate":"b2a","args":[1,1,50]}),
            json!(0),
        ),
        (
            json!({"id":"tiny-compare","gate":"compare","args":[128,8,0,"signed",null]}),
            json!(1),
        ),
        (
            json!({"id":"tiny-phi","gate":"phi-select","args":[1,mask(50),4096]}),
            json!(mask(50)),
        ),
        (
            json!({"id":"tiny-external","gate":"external","args":[-3,8192]}),
            json!(mask(50) - 1),
        ),
    ] {
        // Deliberately corrupt the expected field: execution must ignore it.
        let mut case = case;
        case["expected"] = json!("not an executable input");
        let (result, stats, _) = run_case(&case);
        assert_eq!(result, expected);
        assert_eq!(
            stats.issuances,
            match case["gate"].as_str().unwrap() {
                "external" => 0,
                "phi-select" => 2,
                _ => 1,
            }
        );
    }
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let table =
        lookup::Table::new_zero_extended(50, &[7, 0, mask(50), 4096, 11], source_digest(), &budget)
            .unwrap();
    let mut stats = Stats::default();
    assert_eq!(
        table_values(&table, "tiny-table", 5, &budget, &mut stats),
        [7, 0, mask(50), 4096, 11]
    );
    assert_eq!(stats.issuances, 2);
    assert_eq!(stats.lanes, 5);
    drop(table);
    assert_eq!(budget.used_bytes(), 0);
}
