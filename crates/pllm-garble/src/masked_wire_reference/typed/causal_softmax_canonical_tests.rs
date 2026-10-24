//! Independent D36 metadata goldens; mount ONLY as a test child of
//! causal_softmax_region.rs. Python stdlib SHA-256 over literal byte rules
//! produced the answers, without production digest/quote results as inputs.
//! No native issuance, floor verification, entropy, or source authentication.
//! Identity bytes below are deliberately ordinary TEST labels, not permits.
//! These tests do not certify private inputs, joint privacy, or model quality.
use super::*;
use pllm_core::edge_contract::{ConsumerUse, EdgeIdentity, Shape};

const REF_DOMAIN: &[u8] = b"pllm.sigma.causal-softmax-row.v1";
const REF_EQUATION: &[u8] = b"M=max(active X);E=nExp(M-X);D=sum(E);Z=floor(D/64);R=floor(2^18/Z);P=floor(E*R/4096);inactive=0";
const REF_PAPER: &str = "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf";
const REF_COMMIT: &str = "aa0f2f860266fb79901405ef5e9965b8c9fe2e88";
const REF_TABLES: [&str; 3] = [
    "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
    "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
    "b03af89917afce4cad1e1923bc09064e7d0b19e7f5b906f7694c3a421c3e8f3d",
];
const REF_NUMERIC: &str = "cf7628e36144fd275cf09f4f9b1c4bdf5784d3326e88d9266dcc4b4eeb715cdf";
const REF_INPUT: &str = "af14dc2213e5b46dfd53e25394e581826b439faacb104836bd34bbb8f8ec2898";
const REF_OUTPUT: &str = "3aad7797c798b44ffd897d8f6a326fa243058f239b000491d67b27c7102536d3";
const REF_PUBLIC_OUTPUT: &str = "41b1de10a86386e4c522f63855719bb98c30cb3305170f08a123082176b9fa91";
const REF_DECLARATIONS: [&str; 3] = [
    "246e1b6415ff9832b2d774103c5936b7765730511619b04d86c1c3d966d74785",
    "0ecb2a416fa567acbdbe94f6423bd82c5bc024a7711c7d839e580e21d5ecbfa3",
    "69a86c4ca5b322b4eeacfb4fef8b68b05d6c3798c3dfbedd42bbd6cb1e21451d",
];
const PUBLIC_USES: [&str; 2] = [
    "d606e7fb74797a14f2001f0fe1377c8bf9555c7469395b589f096ac66fefe546",
    "31da9660e8b0ea81b66962b7bf7786c497dc614cd069e9cdb92fcb257de61cbd",
];
const TWO_USES: [&str; 27] = [
    "d606e7fb74797a14f2001f0fe1377c8bf9555c7469395b589f096ac66fefe546",
    "0f754ab2efffe2d1331e0b5ab27ebb9edde5a20cc4b8d4fb2908f6c4f39a6fcd",
    "06f4d74b380bd49573d1c685d7598e13d0c429fc3537392baa1ee8b2631b88e4",
    "6816ecd5f1118f8266d23db39cfabe17d9d8b996bbb8054736e593f9c63ddf9b",
    "3c085a43504adb3aea8b5f72b785ae84d713a820d2cbc1cc54bd57a0db4cd334",
    "3c05831d189aa22388dfcbfe8dd13ddaa8c23ec28fa1ecbed8562938c25d3ad2",
    "586dcf621f2d64af2251101142d200b9ceee99a347f007bffa3a82306955fb19",
    "8f96741717a950ac9afa872f825ede7a348e24bf7e8c90a7a1eac4ad52c850b8",
    "ffe39266e7f914b9dd6154c0085d65f35f9d80454e68d2dfad4242d5fa4409d8",
    "1b4d68cc561444376e702e091dffa810f696d5a573d6aef6393e87f5a6131405",
    "ca2e60107ee2bcd9ae09a87406b7a8ba2be969484255a0a06676e6b7bdef6d35",
    "3c650f8b875b0f7a17e62d5bc416b4a22beba590309c029cbaf88ea0c74ec3b7",
    "f540e75fcc4e0eeb8ea6d0aa093743fc940454139527cc1c62a670766f62fd01",
    "6e7158aca5bb50038230f338af8cc5a10190d7891d6e8c4ea00ebb3f62706f6c",
    "1df93c81c5d0794a250cfd498513084c1f2fd95db4b5b2c0cea856bbd1bb0f92",
    "cc314d9ef1e3442c6379611c7e134eaa19eb7ff31c52707fa5a967948300cde5",
    "da2649669b91b34417dbf6d89dde0b8050f7f266cb2779cc2adf018feefcb90e",
    "22fbf67ca83fe07c931b89175372c7a6e9e2788631ab5a93778be54773fba605",
    "383f881d77cc8a3af13f96f41a0785e619700037a9160c0ed4f6cc1a5b8efbcf",
    "13950d1e74b4271b0f4303b8ea516a76cc4d0728cc2983945833f93af0909219",
    "e38d169db2013927d6dbabf69abac1a3e5fc31a85261f7200a470e954648ce96",
    "ae97326bc76e5bd6b4ec0d02ebde5ea102fcae679529fbbe4c1bfd415be51244",
    "51a3c6fcfe7cb9d74db8cefa15637f75e070c11cb5a20ee558cdf4782902aa92",
    "befef99dbd10c0088ef3548ed924b092b5b6da7be9ce88bd522c364987ac4bfc",
    "2a0f654ca22adce3d12ae454167b2ba55296f8fa2398f53e9ae0fc50afdec58a",
    "db98617c4f3317ecbc309181322e7c2df74df896dac5b072d2f7cb1899e74821",
    "cb5327756aa45a73040b7d222ac86d95c133d5cf1589504893ba07128d1dec3d",
];
const THREE_USES: [&str; 40] = [
    "d606e7fb74797a14f2001f0fe1377c8bf9555c7469395b589f096ac66fefe546",
    "0f754ab2efffe2d1331e0b5ab27ebb9edde5a20cc4b8d4fb2908f6c4f39a6fcd",
    "06f4d74b380bd49573d1c685d7598e13d0c429fc3537392baa1ee8b2631b88e4",
    "6816ecd5f1118f8266d23db39cfabe17d9d8b996bbb8054736e593f9c63ddf9b",
    "3c085a43504adb3aea8b5f72b785ae84d713a820d2cbc1cc54bd57a0db4cd334",
    "3c05831d189aa22388dfcbfe8dd13ddaa8c23ec28fa1ecbed8562938c25d3ad2",
    "586dcf621f2d64af2251101142d200b9ceee99a347f007bffa3a82306955fb19",
    "8f96741717a950ac9afa872f825ede7a348e24bf7e8c90a7a1eac4ad52c850b8",
    "ffe39266e7f914b9dd6154c0085d65f35f9d80454e68d2dfad4242d5fa4409d8",
    "1b4d68cc561444376e702e091dffa810f696d5a573d6aef6393e87f5a6131405",
    "7430fc272ca2ce30d2322ad50d007effbe127ff8ec7e2243f97d357f012a7ccc",
    "77e7f24679b266c7b78251d2d525c128f71a13f86e64a0e9244c7aaa057168c6",
    "547608049585f564ee52be15a8114287f1a01931ec561e90b9d31ea95cd454eb",
    "2a7db1ddb51cad153fdc01bd83fafe79c577decd1cbaea698b97e03c7fffbb1d",
    "18959b1d21a192804caade0a3f0b986d730ac753a9c4808ddc00c07cb7d5ec8c",
    "eb287b32db7ce3bfdb6242a91c512a8d0b4c41e8beecc3ee28a4d03c5995d46c",
    "03ed04f9208fd83b8c5700aad85b120ede864a8a605e82e41c45d454c8fe13b3",
    "4c5ef71fd2deaac5439dde673e7267cde2f86893ead4879affd0bcbf8d32494d",
    "1f0571ef6688813ff3d30c7f39a37f67df6acf6a18c6899b7d26c743fd4b5fd1",
    "3f0b82f8532cd99d2c5f96d45a6deb68406b77269e090c038aef9c8570921ba3",
    "a948ebab985c8c81d3317c7f9dc069bed347b5a81d0a2f132a087fc1564fdcb6",
    "b31cb50996ebbe3ed791cb6a071753ba9dc226eac30344a0d05875b7ddd56e49",
    "cdb66ddd8816af73cda697ff1e63ac2e9ded532b5fcb06227cd3cd7e5296afff",
    "9f19ed5d0d75e248e44019a5e8d8193d7cb8c486870f1e69d410746587d19f25",
    "d107725806f631788ab4cdaf800c8d5e8823c4d58d25f6c288632bc5f99c6226",
    "3de0cbd5c373895cda7616be5fa1cd8dfe4ab8d3225990ae08d4095f16ed01ed",
    "8eda79db7c03a3f011a2d2ee9e7c3703e33bc0f9382e3f51c73e2a860bd7a767",
    "c9746e034c9252823b9cd3dbab1d4e13eec15aafbc89d2cb8ad274fec95ce285",
    "31565cb94f759a1b326fcd66f25ddb1a0941f1efe8c2c11136c85faec323962b",
    "8b9284644cd92471dbd823a1522426ddda8e463ee944aca09978e6f25f4c7d78",
    "256cddf0ad62ef811050466de7b3f1821f034eaa192635610fa1eff409cad728",
    "fde5f9dc40bb52771d48410b440b0a3866e631c04f73fb0fbff83e328268b420",
    "c66ddef0060a26690bee90adef7df53a5800f321392a74fe98b75d202b0b62b6",
    "20d36648df4b0e21e44438d28a9bc35f94ffcf996f66080bb7f71f5ae15d3e7a",
    "9ed315fcb2305bd80d5b8fc93f6ded33055aab0061acbdde3ed996e5284877ca",
    "9c1256d0f1b8717db793f08c9c98e86bf707d43753ef37c188e44ca9fb898927",
    "f54ad21e63f113d13442f95ca240d898263605c499109acb0b43a523456e6d7f",
    "7ac4af4b00e8d4daaeed9fd43819b7e3966de647b5209290a6a2184e6c6ef037",
    "fd514372bd633e09eabefba0a204f6054af483fbb5e0d73ce93dcdad86b0dc6f",
    "bd779426d5dbed10d4937e6b17e71433607bcbb5aab2be9cab6452d859575e56",
];

fn unhex(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64);
    std::array::from_fn(|i| u8::from_str_radix(&value[2 * i..2 * i + 2], 16).unwrap())
}

fn framed(fields: &[&[u8]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for field in fields {
        bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
        bytes.extend_from_slice(field);
    }
    bytes
}

fn fixture(active: &[u8]) -> CausalSoftmaxSpec {
    let source = SourceIdentity::new([1; 32], [2; 32], [3; 32]);
    let descriptor = |wire, representation| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source, [4; 32], [5; 32], [wire; 32], 0x0102_0304_0506_0708),
            Shape::new(&[3]).unwrap(),
            Numeric::new(50, 12, Interpretation::Signed).unwrap(),
            representation,
            None,
        )
        .unwrap()
    };
    CausalSoftmaxSpec::checked(
        descriptor(6, Representation::CommonAdditiveMasked),
        descriptor(
            7,
            if active.len() == 1 {
                Representation::Public
            } else {
                Representation::CommonAdditiveMasked
            },
        ),
        active,
        [9; 32],
        [10; 32],
    )
    .unwrap()
}

fn reference_numeric_bytes() -> Vec<u8> {
    framed(&[
        REF_DOMAIN,
        b"inverse.numeric.v1",
        REF_EQUATION,
        &[1; 32],
        &[2; 32],
        &[3; 32],
        REF_PAPER.as_bytes(),
        REF_COMMIT.as_bytes(),
        &unhex(REF_TABLES[0]),
        &unhex(REF_TABLES[1]),
        &unhex(REF_TABLES[2]),
        &[8, 0, 6, 50, 12],
        b"T[0]=0",
        b"T[z]=floor(2^18/z)",
    ])
}

fn reference_declaration_bytes(active: &[u8], expected_uses: &[&str]) -> Vec<u8> {
    let mut bytes = framed(&[
        REF_DOMAIN,
        b"declaration.v1",
        REF_EQUATION,
        &[1; 32],
        &[2; 32],
        &[3; 32],
        &[4; 32],
        &[8; 32],
        &[9; 32],
        &[10; 32],
        REF_PAPER.as_bytes(),
        REF_COMMIT.as_bytes(),
        &unhex(REF_INPUT),
        &unhex(if active.len() == 1 {
            REF_PUBLIC_OUTPUT
        } else {
            REF_OUTPUT
        }),
        &[3, 0, 0, 0, 0, 0, 0, 0],
        &[active.len() as u8],
        active,
        // Signed i128 LE endpoints -2^37 and +2^37, not native enum layouts.
        &[
            0, 0, 0, 0, 224, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ],
        &[0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[50, 12, 39, 14, 12, 6, 8, 6, 0, 50, 24, 12],
        &unhex(REF_TABLES[0]),
        &unhex(REF_TABLES[1]),
        &unhex(REF_TABLES[2]),
        &unhex(REF_NUMERIC),
    ]);
    for expected in expected_uses {
        bytes.extend(framed(&[&unhex(expected)]));
    }
    bytes
}

#[test]
fn exact_source_table_pins_and_582_byte_numeric_record() {
    assert_eq!(crate::fixed_exp_tables::SOURCE_COMMIT, REF_COMMIT);
    assert_eq!(crate::fixed_exp_tables::PAPER_SHA256, REF_PAPER);
    for (table, expected) in [
        (crate::fixed_exp_tables::Table::LowByte, REF_TABLES[0]),
        (crate::fixed_exp_tables::Table::HighByte, REF_TABLES[1]),
    ] {
        assert_eq!(table.definition().sha256_u64le, expected);
    }
    assert_eq!(fixed_inverse_table::SHA256_U64LE, unhex(REF_TABLES[2]));
    assert_eq!(fixed_inverse_table::GLOBAL_MAX, 262144);
    let bytes = reference_numeric_bytes();
    assert_eq!(bytes.len(), 582);
    assert_eq!(<[u8; 32]>::from(Sha256::digest(bytes)), unhex(REF_NUMERIC));
    assert_eq!(
        records::numeric(
            fixture(&[0, 2]).input.identity().source(),
            REF_TABLES.map(unhex)
        ),
        unhex(REF_NUMERIC)
    );
}

#[test]
fn single_public_and_27_40_ordered_use_prefixes_match_literal_preimages() {
    for (active, expected_uses, length, declaration) in [
        (&[2][..], &PUBLIC_USES[..], 987, REF_DECLARATIONS[0]),
        (&[0, 2][..], &TWO_USES[..], 1988, REF_DECLARATIONS[1]),
        (&[0, 1, 2][..], &THREE_USES[..], 2509, REF_DECLARATIONS[2]),
    ] {
        let s = fixture(active);
        let actual = uses(&s, &child_edges(&s).unwrap()).unwrap();
        assert_eq!(s.outer_uses(), expected_uses.len());
        if active.len() > 1 {
            assert_eq!(expected_uses.len(), 13 * active.len() + 1);
        }
        assert_eq!(s.input.digest(), unhex(REF_INPUT));
        assert_eq!(
            s.output.digest(),
            unhex(if active.len() == 1 {
                REF_PUBLIC_OUTPUT
            } else {
                REF_OUTPUT
            })
        );
        for (i, expected) in expected_uses.iter().enumerate() {
            assert_eq!(
                actual[i].digest(),
                unhex(expected),
                "k={}, use={i}",
                active.len()
            );
            assert_eq!(actual[i].ordinal(), i as u32);
            assert_eq!(actual[i].operand_position(), 0);
            assert_eq!(
                actual[i].consumer(),
                if i + 1 == expected_uses.len() {
                    [10; 32]
                } else {
                    [9; 32]
                }
            );
        }
        let bytes = reference_declaration_bytes(active, expected_uses);
        assert_eq!(bytes.len(), length);
        assert_eq!(<[u8; 32]>::from(Sha256::digest(bytes)), unhex(declaration));
        assert_eq!(records::record_bytes(&s).unwrap(), length);
        assert_eq!(
            records::declaration(
                &s,
                [8; 32],
                REF_TABLES.map(unhex),
                &actual[..s.outer_uses()]
            ),
            unhex(declaration)
        );
    }
}

#[test]
fn k2_six_and_k3_eight_outer_stage_ids_use_frozen_framing_and_u16_le_labels() {
    let two = [
        "e6f7814cc22ce046e0e875e3f82bd5c89e5f01012dcb3750415dbfae881b8c72",
        "3452cab2191e7c83dc887c1ea89fcf8b9985e402a841ea9ff5503023605c0386",
        "694b1f714f1b22817b894a5cebb90dcf8d66f9e6efc66250ca06123f2f4e670e",
        "53d5f8ec9bba9c3cd14d5e5c7be3db8e4c438e0a6f2f203b41a177396ff48573",
        "c2534bea6442cfe7d502f5d239e3efa911fc556af4c2584a9540a32390a66fd5",
        "c467e6c1679d4eb73b7e569c0205c3e0e51dd9f6aa98412b815e5a0765741b5c",
    ];
    let three = [
        "d16892086f9149bd2d3ff3d4a4a00da7d4716488d94e52b1cf591ca16e68a6ad",
        "405492b86f8ae21b77ee00ef46a25381f13e99d1d287fb7ffde2dff1180f9c2c",
        "606e9f622d8460cc52d56dec3df765cc5fcac778472d6ef26fac5872fd1383d6",
        "942cfacc9604eb1c4fa1d054ec024cf3b7cc6f3eb233a04990e07af1a073c496",
        "e0c4609404e31776f80f09b5fa34620f2283bc21829d00e9fdcf58f0afa8533b",
        "ac3bf8208d7b1558400a157aeb450e12565f9fa05f7ddd14c4abff637622cbbb",
        "1cdfe5b404bc91d4a2c81e54194d55f23e74445d9b166ce38ee0e339b92bc623",
        "8668dca522f902a31926d46f0a6bf403c3da3fb62a5b28dfbd76ee0709025946",
    ];
    for (k, declaration, expected) in [
        (2, REF_DECLARATIONS[1], &two[..]),
        (3, REF_DECLARATIONS[2], &three[..]),
    ] {
        // Two max stages/update, then coordinate/inverse/product/Gap.
        // The inherited D29 child keeps its own ORIGINAL eleven contexts.
        assert_eq!(expected.len(), 2 * (k - 1) + 4);
        for (i, expected) in expected.iter().enumerate() {
            let bytes = framed(&[
                REF_DOMAIN,
                b"stage.v1",
                &unhex(declaration),
                &[(i + 1) as u8, 0],
            ]);
            assert_eq!(<[u8; 32]>::from(Sha256::digest(bytes)), unhex(expected));
            assert_eq!(
                records::stage(unhex(declaration), (i + 1) as u16),
                unhex(expected)
            );
        }
    }
}

#[test]
fn unsigned_full_sum_projected_denominator_and_q0_coordinate_are_distinct() {
    for active in [&[0, 2][..], &[0, 1, 2][..]] {
        let s = fixture(active);
        let edges = child_edges(&s).unwrap();
        for (read, bits, scale, expected) in [
            (
                Read::Project,
                50,
                12,
                "7b39146138aa79c887992749cb947641bacc90d9875eab8aec79b1675bf6ec3c",
            ),
            (
                Read::Truncate,
                14,
                12,
                "dc115880fa4117d608e4370efc5778860dd3cdc217073539bf7081797f99a163",
            ),
            (
                Read::Coordinate,
                8,
                0,
                "2ef83af2cfef9c9ee097c94de213d3d6b11ca5e3c8aa0cdf993b3701db222721",
            ),
        ] {
            let edge = edge_for(&s, read, &edges).unwrap();
            assert_eq!(edge.shape().elements(), 1);
            assert_eq!(
                edge.numeric(),
                Numeric::new(bits, scale, Interpretation::Unsigned).unwrap()
            );
            assert_eq!(edge.digest(), unhex(expected));
        }
    }
    // Algebraic witnesses only: NO replacement for a genuine D/range owner.
    // Native TR input is projected common word (r+D) mod 2^14, not the
    // clear denominator. Never apply true-secret bounds to uniform masks.
    let ring50 = (1u64 << 50) - 1;
    let ring14 = (1u64 << 14) - 1;
    for d in [4096u64, 8192, 12288] {
        for r in [0, 8191, 16383, ring50] {
            let full_word = r.wrapping_add(d) & ring50;
            let native_word = full_word & ring14;
            let native_mask = r & ring14;
            assert_eq!(native_word.wrapping_sub(native_mask) & ring14, d);
            assert!((64..=192).contains(&(d >> 6)));
        }
    }
    // 8192 is positive/valid in unsigned14, negative in signed14. Q0
    // coordinate code 64 denotes Q6 denominator 1, not a signed TR result.
    assert_eq!(8192i64 - (1 << 14), -8192);
    assert_eq!(4096u64 >> 6, 64);
}

#[test]
fn ordered_prefix_excludes_padding_and_binds_session_table_operand_and_ordinal() {
    let s = fixture(&[0, 2]);
    let actual = uses(&s, &child_edges(&s).unwrap()).unwrap();
    let n = 27;
    let baseline = unhex(REF_DECLARATIONS[1]);
    let record =
        |values: &[ConsumerUse]| records::declaration(&s, [8; 32], REF_TABLES.map(unhex), values);
    assert_eq!(record(&actual[..n]), baseline);
    assert_ne!(record(&actual), baseline);
    let mut changed = actual;
    for value in &mut changed[n..] {
        *value = ConsumerUse::new(s.input, [0xff; 32], u16::MAX, u32::MAX);
    }
    assert_eq!(record(&changed[..n]), baseline);
    for (consumer, operand, ordinal) in [([0xee; 32], 0, 1), ([9; 32], 1, 1), ([9; 32], 0, 2)] {
        let mut changed = actual;
        changed[1] = ConsumerUse::new(actual[1].edge(), consumer, operand, ordinal);
        assert_ne!(record(&changed[..n]), baseline);
    }
    let mut changed = actual;
    changed.swap(1, 2);
    assert_ne!(record(&changed[..n]), baseline);
    assert_ne!(
        records::declaration(&s, [0; 32], REF_TABLES.map(unhex), &actual[..n]),
        baseline
    );
    let mut hashes = REF_TABLES.map(unhex);
    hashes[2][0] ^= 1;
    assert_ne!(
        records::declaration(&s, [8; 32], hashes, &actual[..n]),
        baseline
    );
    // Hash sensitivity is declaration integrity, NOT a signature/capability.
    // No fabricated Binding, table Verified, material, or lease is issued here.
}
