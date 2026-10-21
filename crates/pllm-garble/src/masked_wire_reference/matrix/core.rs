//! Shared bounded matrix arithmetic, independent of ownership and framing.
use super::{product, split, Budget, Error, Spec, MAX_COMPLETE_MACS, MAX_ELEMENTS};
use crate::shared_allocation::allocate;
use pllm_core::streamed_matrix::{accumulate_resident, MatrixGeometry};
use zeroize::Zeroizing;

pub(super) fn dimensions(spec: Spec) -> Result<(usize, usize, usize, usize), Error> {
    if !(1..=64).contains(&spec.bits) {
        return Err(Error::Domain);
    }
    let a = spec.rows.checked_mul(spec.inner).ok_or(Error::Resource)?;
    let b = spec
        .inner
        .checked_mul(spec.columns)
        .ok_or(Error::Resource)?;
    let c = spec.rows.checked_mul(spec.columns).ok_or(Error::Resource)?;
    if [a, b, c].iter().any(|n| !(1..=MAX_ELEMENTS).contains(n)) {
        return Err(Error::Resource);
    }
    let d = a.checked_mul(spec.columns).ok_or(Error::Resource)?;
    if d.checked_mul(6).ok_or(Error::Resource)? > MAX_COMPLETE_MACS {
        return Err(Error::Resource);
    }
    geometry(spec)?;
    Ok((a, b, c, d))
}
pub(super) fn geometry(spec: Spec) -> Result<MatrixGeometry, Error> {
    MatrixGeometry::new(spec.rows, spec.inner, spec.columns, spec.layout)
        .map_err(|_| Error::Resource)
}
pub(super) fn mask(bits: u8) -> u64 {
    u64::MAX >> (64 - bits)
}
pub(super) fn validate(values: &[u64], len: usize, bits: u8) -> Result<(), Error> {
    if values.len() != len || values.iter().any(|v| v & !mask(bits) != 0) {
        return Err(Error::Domain);
    }
    Ok(())
}
pub(super) fn zeros(len: usize, budget: &Budget) -> Result<Zeroizing<Vec<u64>>, Error> {
    let mut v = allocate(len, budget)?;
    v.resize(len, 0);
    Ok(Zeroizing::new(v))
}
pub(super) struct Material {
    pub(super) a: Zeroizing<Vec<u64>>,
    pub(super) b: Zeroizing<Vec<u64>>,
    pub(super) c: Zeroizing<Vec<u64>>,
}
pub(super) fn deal(
    spec: Spec,
    [rx, rw, ry]: [&[u64]; 3],
    budget: &Budget,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<[Material; 2], Error> {
    let (a, b, c, _) = dimensions(spec)?;
    validate(rx, a, spec.bits)?;
    validate(rw, b, spec.bits)?;
    validate(ry, c, spec.bits)?;
    let mut materials = [
        Material {
            a: zeros(a, budget)?,
            b: zeros(b, budget)?,
            c: zeros(c, budget)?,
        },
        Material {
            a: zeros(a, budget)?,
            b: zeros(b, budget)?,
            c: zeros(c, budget)?,
        },
    ];
    let mut correction = zeros(c, budget)?;
    correction.copy_from_slice(ry);
    accumulate_resident(
        rw,
        &[rx],
        &mut [&mut correction],
        geometry(spec)?,
        cancelled,
    )
    .map_err(|_| Error::Consumed)?;
    for (kind, values) in [(0, rx), (1, rw), (2, &*correction)] {
        for (i, &value) in values.iter().enumerate() {
            if cancelled() {
                return Err(Error::Consumed);
            }
            let value = Zeroizing::new(if kind < 2 {
                product::negative_mask(value)
            } else {
                value
            });
            let shares = Zeroizing::new(split(*value, spec.bits, fill)?);
            for (party, material) in materials.iter_mut().enumerate() {
                let output = match kind {
                    0 => &mut material.a,
                    1 => &mut material.b,
                    _ => &mut material.c,
                };
                output[i] = shares[party];
            }
        }
    }
    Ok(materials)
}
pub(super) fn evaluate(
    spec: Spec,
    role: u8,
    material: &Material,
    xhat: &[u64],
    what: &[u64],
    budget: &Budget,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<Zeroizing<Vec<u64>>, Error> {
    let (a, b, c, _) = dimensions(spec)?;
    if role > 1 {
        return Err(Error::Context);
    }
    validate(xhat, a, spec.bits)?;
    validate(what, b, spec.bits)?;
    validate(&material.a, a, spec.bits)?;
    validate(&material.b, b, spec.bits)?;
    validate(&material.c, c, spec.bits)?;
    let mut output = zeros(c, budget)?;
    output.copy_from_slice(&material.c);
    let geometry = geometry(spec)?;
    accumulate_resident(
        what,
        &[&material.a],
        &mut [&mut output],
        geometry,
        cancelled,
    )
    .map_err(|_| Error::Consumed)?;
    accumulate_resident(
        &material.b,
        &[xhat],
        &mut [&mut output],
        geometry,
        cancelled,
    )
    .map_err(|_| Error::Consumed)?;
    if role == 1 {
        accumulate_resident(what, &[xhat], &mut [&mut output], geometry, cancelled)
            .map_err(|_| Error::Consumed)?;
    }
    for value in output.iter_mut() {
        *value &= mask(spec.bits);
    }
    if cancelled() {
        return Err(Error::Consumed);
    }
    Ok(output)
}
