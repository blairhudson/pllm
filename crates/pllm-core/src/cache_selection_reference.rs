//! Bounded plaintext KV retention/cluster-selection oracle. No protected top-k.
//! Independently follows MPCache sections 4.2–4.4, equations (2) and (3).
//! Head aggregation and retention fractions belong to the public caller policy.

pub const MAX_ROWS: usize = 8192;
pub const MAX_FEATURES: usize = 1024;

fn finite(values: &[f32]) -> Result<(), &'static str> {
    if values.iter().any(|v| !v.is_finite()) {
        Err("non-finite selection input")
    } else {
        Ok(())
    }
}

fn ranked(scores: &[f64], keep: usize) -> Vec<usize> {
    let mut order: Vec<_> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order.truncate(keep);
    order.sort_unstable();
    order
}

/// Stable static top-k with an explicitly retained recent suffix. Indices are
/// strictly increasing original token positions, never importance-rank order.
pub fn retain(scores: &[f32], keep: usize, recent: usize) -> Result<Vec<usize>, &'static str> {
    if scores.is_empty()
        || scores.len() > MAX_ROWS
        || keep == 0
        || keep > scores.len()
        || recent > keep
    {
        return Err("invalid retention bounds");
    }
    finite(scores)?;
    let end = scores.len() - recent;
    let scores: Vec<_> = scores[..end].iter().map(|&v| f64::from(v)).collect();
    let mut selected = ranked(&scores, keep - recent);
    selected.extend(end..end + recent);
    Ok(selected)
}

/// Scores contiguous original-position clusters over a retained candidate set.
/// Alpha-weighted extrema approximate equation (3), NOT an upper bound.
/// `alpha=None` uses the sign-aware equation (2) upper bound instead.
pub fn clusters(
    keys: &[f32],
    query: &[f32],
    candidates: &[usize],
    cluster_rows: usize,
    keep_clusters: usize,
    alpha: Option<f32>,
) -> Result<(Vec<usize>, Vec<f64>), &'static str> {
    let width = query.len();
    if width == 0 || width > MAX_FEATURES || keys.is_empty() || keys.len() % width != 0 {
        return Err("invalid key/query shape");
    }
    let rows = keys.len() / width;
    if rows > MAX_ROWS
        || candidates.is_empty()
        || cluster_rows == 0
        || cluster_rows > MAX_ROWS
        || candidates.windows(2).any(|v| v[0] >= v[1])
        || *candidates.last().unwrap() >= rows
        || alpha.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err("invalid cluster domain");
    }
    finite(keys)?;
    finite(query)?;
    let groups: Vec<_> = candidates
        .chunk_by(|a, b| a / cluster_rows == b / cluster_rows)
        .collect();
    if keep_clusters == 0 || keep_clusters > groups.len() {
        return Err("invalid cluster count");
    }
    let scores: Vec<f64> = groups
        .iter()
        .map(|group| {
            (0..width)
                .map(|col| {
                    let lo = group
                        .iter()
                        .map(|&row| keys[row * width + col])
                        .fold(f32::INFINITY, f32::min);
                    let hi = group
                        .iter()
                        .map(|&row| keys[row * width + col])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let q = f64::from(query[col]);
                    match alpha {
                        Some(a) => {
                            q * (f64::from(a) * f64::from(hi)
                                + (1.0 - f64::from(a)) * f64::from(lo))
                        }
                        None => q * f64::from(if q >= 0.0 { hi } else { lo }),
                    }
                })
                .sum()
        })
        .collect();
    let selected = ranked(&scores, keep_clusters)
        .into_iter()
        .flat_map(|i| groups[i].iter().copied())
        .collect();
    Ok((selected, scores))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn static_order_recent_suffix_and_ties() {
        assert_eq!(retain(&[8., 8., 7., 0., 0.], 3, 1).unwrap(), [0, 1, 4]);
        assert_eq!(
            retain(&[8., 8., 7., 0., 0.], 5, 1).unwrap(),
            [0, 1, 2, 3, 4]
        );
        assert!(retain(&[f32::NAN], 1, 0).is_err());
        assert!(retain(&[1., 2.], 1, 2).is_err());
    }
    #[test]
    fn bound_contains_every_dot_but_affine_summary_does_not() {
        let keys = [-2., 2., 0., 0., 2., -2., 1., 1.];
        for query in [[-1., 0.], [1., -3.], [0.5, 0.25]] {
            let (_, bounds) = clusters(&keys, &query, &[0, 1, 2, 3], 2, 2, None).unwrap();
            for (row, key) in keys.chunks(2).enumerate() {
                let dot: f64 = key
                    .iter()
                    .zip(query)
                    .map(|(&k, q)| f64::from(k) * f64::from(q))
                    .sum();
                assert!(dot <= bounds[row / 2]);
            }
        }
        let (_, scores) = clusters(&[-2., 2.], &[-1.], &[0, 1], 2, 1, Some(0.6)).unwrap();
        assert!(scores[0] < 0.0); // max dot is +2: approximation cannot certify pruning.
    }
    #[test]
    fn hierarchical_selection_preserves_original_clusters_and_all_row_control() {
        let keys: Vec<_> = (0..13).map(|i| i as f32).collect();
        let rows: Vec<_> = (0..13).collect();
        let (coarse, _) = clusters(&keys, &[1.], &rows, 8, 1, Some(0.6)).unwrap();
        assert_eq!(coarse, [8, 9, 10, 11, 12]);
        let (fine, _) = clusters(&keys, &[1.], &coarse, 4, 1, Some(0.6)).unwrap();
        assert_eq!(fine, [12]);
        assert_eq!(
            clusters(&keys, &[1.], &rows, 4, 4, Some(0.6)).unwrap().0,
            rows
        );
        assert!(clusters(&keys, &[1.], &[1, 1], 4, 1, None).is_err());
        assert!(clusters(&keys, &[1.], &[13], 4, 1, None).is_err());
        assert!(clusters(&keys, &[f32::INFINITY], &[1], 4, 1, None).is_err());
    }
    #[test]
    fn same_public_shapes_can_reveal_query_through_gather_indices() {
        let keys = [-2., -1., 1., 2.];
        let a = clusters(&keys, &[1.], &[0, 1, 2, 3], 2, 1, Some(0.6))
            .unwrap()
            .0;
        let b = clusters(&keys, &[-1.], &[0, 1, 2, 3], 2, 1, Some(0.6))
            .unwrap()
            .0;
        assert_ne!(a, b); // Publicly indexed remote gathers are not a protected protocol.
    }
}
