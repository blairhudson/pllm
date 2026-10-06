//! Bounded JSON-line bridge for public checkpoint diagnostics.
use pllm_core::polynomial_softmax_reference::{evaluate, Method};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, BufRead, Read, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    method: String,
    rows: Vec<Vec<f32>>,
}

fn main() -> Result<(), String> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    loop {
        let mut bytes = Vec::new();
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
        if request.rows.is_empty() || request.rows.len() > 8192 {
            return Err("too many softmax rows".into());
        }
        let method = match request.method.as_str() {
            "nexus_exp" => Method::RepeatedSquaring,
            "thor_exp" => Method::NormalizedSquaring,
            _ => return Err("unknown approximation".into()),
        };
        // Domain failure returns no partial tensor, allowing the diagnostic to
        // retain a rejection without pretending an ordinary softmax was used.
        let answer = match request
            .rows
            .iter()
            .map(|row| evaluate(row, method))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(rows) => json!({"rows": rows, "error": null}),
            Err(error) => json!({"rows": null, "error": error}),
        };
        writeln!(output, "{answer}").map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}
