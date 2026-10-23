//! Independent fixed-pin v1 serialization/frame goldens. Mount as a child of
//! nexp_region_tests.rs, not production. Python stdlib SHA-256 over literal byte
//! rules generated these answers; no production digest/quote supplied expected
//! values. Checkpoint/profile/producer/namespace/mask IDs below are TEST labels,
//! not authenticated sources or issued capabilities. Only the final test uses
//! actual original region issuance. No inherited/compiler/SDK/privacy authority.
use super::*;

const IMPLEMENTATION: &str = "fac97122fa3158260d7dbd128e3fe85502e7141f73b437e832ac9e328d95500a";
const TABLES: [&str; 2] = [
    "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
    "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
];
const DECLARATION: &str = "64d5999c8cc0c9e3cc675d206d49773e0ee7ad2a74ce006e3d1c6febab46afc5";
const COORDINATES: [&str; 2] = [
    "bb87b7f12b5d10d68aecd6984bf6437fbf0a56ad4cd6e5f081162a7c660199f6",
    "97ae7110c65503039fe284ce1554ac45278d0da28b0b117b5d1bf7d2b36baf6a",
];
const EDGES: [&str; 14] = [
    "b883919c2e87b81b79b313d2b3a48593be1ab19fd54e7f35a7364fd8c00448a6",
    "3cd92ca5b967b2f6af6d275da1ee7c35a9346397dd70e2259591642745303c7b",
    "a7836bb6d0da51234d09d6f2dc0f965f6a4a991fe9acb8cacc55c06e7ddd3bf0",
    "ae5e71906716223216f2eac718e06a705fc4a86cbd9401cf431feb624bb1b9b9",
    "b129c4c86bf544d8a9d1c85d47b0da0f8f6ff1f75d3822f41d2ec8ede19f3112",
    "ead8c81c868ed2b88c6644908a947db1149a0e2cbc76147136bb0d57f7369f23",
    "45c354b288d21a2f66a97d31766f5c7b06663e6ca4591eabcc21f55a37efa896",
    "1b8664786adc93e4249a083acc9747197fb91daaa8df683750945124307b6d9b",
    "d0ebdfbf746d11ebbfc92259ed2399941bf55b09057ddc5197ef3f55d8fb92ba",
    "07efcd95286d358fe4297eed2407e271a14817c0fd55b79343e20585a71d6c6b",
    "4f6e45cf59595357ab66cd0ef51ff3eba86f6262e8233b338a68691733aedfff",
    "c1932f23ee9a02e62d621f9225c6ec5a9bc2665747711fdfc6bb45dd37d316a7",
    "927838a9f4c4c5afdc64c117b8e2829c622d5b866531fbae461f47446943875b",
    "9ad11886cc6a32b85680fc17b54b4e45faa90ae49f7650d954aa5912f2b8aa34",
];
const USE_DIGESTS: [&str; 18] = [
    "4d01cd9e6eaa51d43078bb1e2ca88103372963ded57fae2262fbd85ecd5066f7",
    "3f847f1cae1f8429d464fdced2f4abd93cacc602a1a3e4403e8253553bbd8216",
    "a5639580944eaf4b9762f875f3be9a8ce95bc75df352be31340d30e65042d6bd",
    "7d282e3db0726c3d3837fd711f180eee62f958f10613ee18bfc91573070623f3",
    "51b05d304401fb72f02a24d2325836ff344a234d7de46b279950340fc8ce3253",
    "aa4588556716791905f7ca79c47a6b10bbc4b342450640e72bc139b72d9cb2b7",
    "a46706c680e788b9daf464ae6ff0ae830f62de8b940c0e77f09287869d9d50a7",
    "87394675d8d8bbb2f75add4a45b75bcf78ccda69ebfafafed70dc21af4515af7",
    "5dcdd0b1afbaf150245faca1438c9cb5bd036db9f09959bc89c8fe15fd5d05b5",
    "7f1c564f9823f9f33669084e3c55d2a1980ef6ac6552aa6b765f202fe134fefd",
    "67e97e5c93a13517cc8801a3d75821a83c58f4e960190fbec24fec126984f721",
    "cecbd2de00d5a483b9799148e54b8bab149e60ba725ea8a73b24f91b68af33f4",
    "a920a783fa77fdebb5cab152fce19d4d068f8249fd4f2c43663252bf28dc859c",
    "1f3c6d692539be9938f64a122b4a69fd8908c32c97de9de724d1783309d538c2",
    "7aec50e0d1c826668300189fc702a3ba9918ffd5958668f83d2ea0962539991b",
    "bf10975048d8ead0f536b430546ba2864331e0f1f06d21497112978101c672d7",
    "1b2c7cc757e15f1e278436e87bea4285365c703db49d8da905d5d2012032b391",
    "e955b9fbe87efd9adfdffb6ba2a74b265a28a30e69c04e82b83eeba06dd7a076",
];
const LEDGERS: [&str; 3] = [
    "6f7d61f7a45773eada80b13650090c82c31b29eef27376a6a6d99403d6eeec9b",
    "ef4a73315acae12f7b7fee86ba8a19590b46bac097cbd0f9ca74d2a7a2448a16",
    "d68e40f0ea3e45dea00d3e162e54c4049f7ab11d65ec5e01bb28f7773225dc4c",
];
const REGION: &str = "013d81fa1fd4d97f990c411ba799ee13b81ea58235ea30f05a0eb695e2d7d827";
const STAGE_IDS: [&str; 11] = [
    "2c51d6715c4da184b075525bf22cf81d3b1a19030891456ee9ea5bdb1f7c7b18",
    "b3af06ba433c4a439cf698d46fe764dc332a02f2bbde642903040d6aade2508d",
    "7a86f8aebf93fbb7c504f3b1d28718847332ab04caf64285cd75f8d5072ee0ce",
    "3dca11c89fd1ce49a0608f2b262492f329ad912db736a05d9de2c70c034c2d07",
    "af5b3872bca2bbc30059f967069d03eac8214f6069ceea8f4e9b612e34c7a95e",
    "7c26ba4317a14b68b8283180ccebd6f9936a8190bcd83b8ae6e1103cdd026cc9",
    "81182a649f48cc22709f0381694ae8c5aa1b897725e7b85635ea417535af53e7",
    "a3453ac3f76702cf0f2f697c61eed4f7b6318f986873a08a34e14401b32caf89",
    "0afadf53f4f7181117bcc9f274703bff85dc7b368b3bdf62f6c6eeeec7e8de22",
    "8960ea12abdf2940129a72a461456f8701b78c0fb872e0580b2e1ffffc14a8c5",
    "e361886f4353c0ec2dc8bdc0238146fd0682dfe73fd6bf69842f2390eddd8be0",
];

fn fixed_spec() -> Spec {
    spec(
        SourceIdentity::new([1; 32], [2; 32], golden(IMPLEMENTATION)),
        2,
    )
}

#[test]
fn real_pins_14_edges_18_uses_and_exact_2037_byte_v1_preimage() {
    assert_eq!(approved_implementation(), golden(IMPLEMENTATION));
    assert_eq!(
        tables::SOURCE_COMMIT,
        "aa0f2f860266fb79901405ef5e9965b8c9fe2e88"
    );
    assert_eq!(
        tables::PAPER_SHA256,
        "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf"
    );
    let s = fixed_spec();
    let uses = usage(&s);
    for (edge, expected) in s.edges.iter().zip(EDGES) {
        assert_eq!(edge.digest(), golden(expected));
    }
    for (use_, expected) in uses.iter().zip(USE_DIGESTS) {
        assert_eq!(use_.digest(), golden(expected));
    }
    let mut bytes = framed(&[
        b"pllm.sigma.nexp-region.v1",
        b"declaration.v1",
        b"floor(E0[min(X,65535)%256]*E1[min(X,65535)/256]/4096)",
        b"f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf",
        b"aa0f2f860266fb79901405ef5e9965b8c9fe2e88",
        &[1; 32],
        &[2; 32],
        &golden(IMPLEMENTATION),
        &[4; 32],
        &[92; 32],
        &[90; 32],
        &[91; 32],
        &[0; 16],
        &[0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[0; 16],
        &[1, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[0, 16, 0, 0, 0, 0, 0, 0],
        &golden(TABLES[0]),
        &golden(TABLES[1]),
        &[12, 4, 50, 12, 16, 8, 0, 50, 24],
        &[2, 0, 0, 0, 0, 0, 0, 0],
        &[7, 0, 0, 0, 0, 0, 0, 0],
    ]);
    for expected in EDGES.into_iter().chain(USE_DIGESTS) {
        bytes.extend(framed(&[&golden(expected)]));
    }
    assert_eq!(bytes.len(), 2037);
    assert_eq!(<[u8; 32]>::from(Sha256::digest(bytes)), golden(DECLARATION));
    assert_eq!(
        spec_digest(&s, TABLES.map(golden), [92; 32], &uses),
        golden(DECLARATION)
    );
    for (i, expected) in COORDINATES.into_iter().enumerate() {
        assert_eq!(coordinate(&s, TABLES.map(golden), i), golden(expected));
    }
}

#[test]
fn runtime_bindings_all_three_role_ledgers_and_11_stage_ids_match_literals() {
    // Serialization data ONLY; these Binding literals cannot escape this test as
    // material, Verified, an original lease, or an inherited capability.
    let s = fixed_spec();
    let intervals = [
        (0, 1i128 << 38),
        (0, 1i128 << 38),
        (-65536, (1i128 << 38) - 65536),
        (0, 2),
        (0, 2),
        (0, 65536),
        (0, 65536),
        (0, 65536),
        (0, 256),
        (0, 256),
        (0, 4097),
        (0, 4097),
        (0, (1i128 << 24) + 1),
        (0, 4097),
    ];
    let bindings: [Binding; 14] = std::array::from_fn(|i| Binding {
        descriptor: s.edges[i],
        session: [92; 32],
        namespace: [6; 32],
        issuance: i as u64,
        mask_id: [0x70 + i as u8; 32],
        consumer: if i == 13 { s.next } else { s.operation },
        interval: Some(Interval {
            lower: intervals[i].0,
            upper: intervals[i].1,
        }),
    });
    for (i, expected) in [
        (
            0,
            "80fc7c43f9d63632759f1fe92f6101a3483bc5de354329bddc99561ad92ff969",
        ),
        (
            12,
            "1792da7f1315b7d1d8a78227c599ba69f3c6caf8f31dd8944a5de547b483d45e",
        ),
        (
            13,
            "a022c57130393aff52f49f7677a60b206d2df77cf26356d27539bca632298eb6",
        ),
    ] {
        assert_eq!(bindings[i].digest(), golden(expected));
    }
    let coords = COORDINATES.map(golden);
    for (role, expected) in LEDGERS.into_iter().enumerate() {
        let ledger = internal_use::Ledger {
            uses: usage(&s),
            bindings: MAP.map(|i| bindings[i]),
            role: role as u8,
            next: 0,
        };
        assert_eq!(ledger_digest(coords, &bindings, &ledger), golden(expected));
    }
    let region = region_digest(
        golden(DECLARATION),
        &bindings,
        LEDGERS.map(golden),
        &[0xc0; 32],
    );
    assert_eq!(region, golden(REGION));
    for (index, expected) in STAGE_IDS.into_iter().enumerate() {
        assert_eq!(stage(region, (index + 1) as u16), golden(expected));
    }
}

#[test]
fn original_issued_frames_follow_literal_11_exchange_contract_and_18_use_handoff() {
    // Actual issuance checks all 512 Rust floors; labels/hashes above are NOT
    // injected as certificates. Retains source's upward 1279 -> 1280 carry step.
    let (mask, coordinator, mut roles, _) = issue(&[1279, 1280]);
    {
        let state = roles[0].slot.state.lock().unwrap();
        assert_eq!(state.tables.hashes, TABLES.map(golden));
        assert_eq!(
            state.tables.declaration.source.implementation(),
            golden(IMPLEMENTATION)
        );
    }
    let frames = complete(&mut roles);
    let layouts: [(&[u8; 8], u8, usize, &[u8]); 11] = [
        (b"PLLMSA01", 1, 76, &[1]),
        (b"PLLMSL01", 1, 76, &[16]),
        (b"PLLMSA01", 0, 76, &[1]),
        (b"PLLMSA01", 1, 76, &[8]),
        (b"PLLMLU01", 0, 80, &[50, 1]),
        (b"PLLMSL01", 1, 76, &[50]),
        (b"PLLMLU01", 0, 80, &[50, 1]),
        (b"PLLMSL01", 1, 76, &[50]),
        (b"PLLMSP01", 1, 76, &[50]),
        (b"PLLMSA01", 0, 76, &[1]),
        (b"PLLMSA01", 1, 76, &[50]),
    ];
    assert_eq!(frames.len(), 11);
    let mut total = 0;
    for (pair, (magic, phase, header, bits)) in frames.iter().zip(layouts) {
        // Native contexts and common-input digests agree between roles; these
        // digests are bindings, NOT authenticated transport or signatures.
        assert_eq!(
            &pair[0].bytes().unwrap()[8..72],
            &pair[1].bytes().unwrap()[8..72]
        );
        for (role, frame) in pair.iter().enumerate() {
            let bytes = frame.bytes().unwrap();
            assert_eq!(&bytes[..8], magic);
            assert_eq!(&bytes[72..76], &[role as u8, phase, 2, 0]);
            assert_eq!(bytes.len(), header + 16 * bits.len());
            if header == 80 {
                assert_eq!(&bytes[76..80], &[8, 50, 0, 0]);
            }
            for (word, chunk) in bytes[header..].chunks_exact(8).enumerate() {
                assert!(
                    u64::from_le_bytes(chunk.try_into().unwrap()) < 1u64 << bits[word % bits.len()]
                );
            }
            total += bytes.len();
        }
    }
    assert_eq!(total, 2104); // 1688 + 208*2, both directions, not quote-self.
    let [left, right] = roles;
    let outputs = [left.finish().unwrap(), right.finish().unwrap()];
    for (role, output) in outputs.iter().enumerate() {
        let state = output.slot.state.lock().unwrap();
        assert_eq!(state.ledger.next, 18);
        assert_eq!(state.ledger.role, role as u8);
        for (use_, expected) in state.ledger.uses.iter().zip(USE_DIGESTS) {
            assert_eq!(use_.digest(), golden(expected));
        }
        let words = state.output.as_ref().unwrap().backing.values();
        let masks = mask.native.backing.values();
        for (i, expected) in [2995, 2996].into_iter().enumerate() {
            assert_eq!(
                words[i].wrapping_sub(masks[i]) & ((1u64 << 50) - 1),
                expected
            );
        }
    }
    drop(coordinator);
    assert!(outputs.iter().all(|output| output.descriptor().is_err()));
    assert!(frames.iter().flatten().all(|frame| frame.bytes().is_err()));
}
