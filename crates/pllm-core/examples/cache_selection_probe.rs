//! Persistent, bounded JSON-line numeric oracle for public diagnostic fixtures.
use pllm_core::cache_selection_reference::{clusters, retain};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, BufRead, Read, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    scores: Option<Vec<f32>>,
    keep: usize,
    recent: Option<usize>,
    keys: Option<Vec<f32>>,
    query: Option<Vec<f32>>,
    candidates: Option<Vec<usize>>,
    cluster_rows: Option<usize>,
    alpha: Option<f32>,
}
fn main() -> Result<(), String> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    loop {
        let mut bytes = Vec::new();
        // Preflight wire input before JSON materializes its numeric arrays.
        let count = (&mut input)
            .take(16 << 20)
            .read_until(b'\n', &mut bytes)
            .map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        if bytes.last() != Some(&b'\n') {
            return Err("oversized or incomplete oracle frame".into());
        }
        let request: Request = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let answer = if let Some(scores) = request.scores {
            json!({"indices": retain(&scores, request.keep, request.recent.unwrap_or(0))?})
        } else {
            let (indices, scores) = clusters(
                &request.keys.ok_or("missing keys")?,
                &request.query.ok_or("missing query")?,
                &request.candidates.ok_or("missing candidates")?,
                request.cluster_rows.ok_or("missing cluster width")?,
                request.keep,
                request.alpha,
            )?;
            json!({"indices": indices, "scores": scores})
        };
        writeln!(output, "{answer}").map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}
