/// Add equal-length tensors in the wrap32 ring.
pub fn add_wrap32(left: &[u32], right: &[u32]) -> Result<Vec<u32>, String> {
    if left.len() != right.len() {
        return Err(format!(
            "wrap32 addition requires equal lengths, received {} and {}",
            left.len(),
            right.len()
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left.wrapping_add(*right))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::add_wrap32;

    #[test]
    fn adds_in_the_wrap32_ring() {
        assert_eq!(add_wrap32(&[u32::MAX, 2], &[2, 3]).unwrap(), vec![1, 5]);
    }

    #[test]
    fn rejects_different_lengths() {
        assert!(add_wrap32(&[1], &[1, 2]).is_err());
    }
}
