//! Allocation-free-of-HE-objects admission probe for one frozen Poulpy profile.
//!
//! This profile is provably over budget before context/key/scratch allocation.
//! Consequently there is no evaluation bypass, secret, ciphertext, or fake
//! refresh path here. Public upstream layout arithmetic independently checks
//! the Python/source-derived lower bound. Build-only work does not run this.

use poulpy_ckks::presets::bootstrapping::n16_d35_k600_p19_c2s;
use poulpy_core::layouts::{GGLWEInfos, LWEInfos};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{Backend, PrepareHint},
};

type Cpu = NTT4x30Portable;
const CEILING: usize = 8 * 1024 * 1024 * 1024;

#[derive(Debug)]
struct Admission {
    automorphisms: usize,
    raw_one: usize,
    prepared_one: usize,
    raw_all_keys: usize,
    prepared_all_keys: usize,
    galois_elements: Vec<i64>,
}

fn bytes<K: GGLWEInfos>(key: &K, prepared: bool) -> usize {
    let shape = (
        key.n().as_usize(),
        key.dnum().as_usize(),
        key.rank_in().as_usize(),
        key.rank_out().as_usize() + 1,
        key.size(),
    );
    if prepared {
        Cpu::bytes_of_vmp_pmat(
            shape.0,
            shape.1,
            shape.2,
            shape.3,
            shape.4,
            PrepareHint::Reuse,
        )
    } else {
        Cpu::bytes_of_mat_znx(shape.0, shape.1, shape.2, shape.3, shape.4)
    }
}

fn admission() -> Result<Admission, String> {
    let p = n16_d35_k600_p19_c2s().map_err(|e| e.to_string())?;
    if (
        p.n(),
        p.base2k(),
        p.input_k(),
        p.bootstrap_k(),
        p.output_k(),
        p.log_delta(),
    ) != (65536, 52, 40, 1427, 600, 35)
    {
        return Err("pinned preset shape drift".into());
    }
    if <<Cpu as Backend>::TaskExecutor as TaskExecutor>::IS_PARALLEL
        || <<Cpu as Backend>::TaskExecutor as TaskExecutor>::max_parallelism() != 1
    {
        return Err("backend is not serial".into());
    }
    let galois_elements = p.plan().galois_elements(p.log_n(), (2 * p.n()) as i64);
    if galois_elements.len() != 33 || !galois_elements.contains(&-1) {
        return Err("pinned rotation/conjugation inventory drift".into());
    }
    let k = p.keys_layout();
    let switches = k
        .encapsulation
        .as_ref()
        .ok_or("missing encapsulation layout")?;
    let raw_one = bytes(&k.automorphism_key, false);
    let prepared_one = bytes(&k.automorphism_key, true);
    let all = |prepared| {
        galois_elements.len() * bytes(&k.automorphism_key, prepared)
            + bytes(&k.tensor_key, prepared)
            + bytes(&switches.dense_to_sparse, prepared)
            + bytes(&switches.sparse_to_dense, prepared)
    };
    Ok(Admission {
        automorphisms: galois_elements.len(),
        raw_one,
        prepared_one,
        raw_all_keys: all(false),
        prepared_all_keys: all(true),
        galois_elements,
    })
}

fn main() {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["admission-report"] {
        eprintln!(
            "only admission-report is implemented; encrypted evaluation is blocked by the frozen 8-GiB contract"
        );
        std::process::exit(78);
    }
    match admission() {
        Ok(a) => {
            println!(
                concat!(
                    "{{\"schema\":\"pllm.poulpy-admission.v1\",",
                    "\"status\":\"blocked_before_keygen\",\"ceiling_bytes\":{},",
                    "\"automorphism_count\":{},\"one_raw_key_bytes\":{},",
                    "\"one_prepared_key_bytes\":{},\"raw_all_key_payload_bytes\":{},",
                    "\"prepared_all_key_payload_bytes\":{},\"all_key_preparation_overlap_bytes\":{},",
                    "\"galois_elements\":{:?},\"full_owner_bound_bytes\":null,",
                    "\"encrypted_refreshes\":0,\"secrets_created\":0,",
                    "\"parameter_qualification\":\"unqualified\"}}"
                ),
                CEILING,
                a.automorphisms,
                a.raw_one,
                a.prepared_one,
                a.raw_all_keys,
                a.prepared_all_keys,
                a.raw_all_keys + a.prepared_all_keys,
                a.galois_elements
            );
            // An unexpected small result cannot silently admit an unqualified
            // parameter set or fill missing owner categories with zero.
            std::process::exit(78);
        }
        Err(e) => {
            eprintln!("pinned admission contract mismatch: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile with --no-run once build is admitted. These exercise upstream
    // public layouts without creating a Module, key, or ciphertext.
    #[test]
    fn actual_pinned_layouts_reject_eight_gib_before_he_allocation() {
        let a = admission().unwrap();
        assert_eq!(a.raw_one, 242_221_056);
        assert_eq!(a.prepared_one, 968_884_224);
        assert_eq!(a.automorphisms * a.prepared_one, 31_973_179_392);
        assert!(a.prepared_all_keys > CEILING);
    }
}
