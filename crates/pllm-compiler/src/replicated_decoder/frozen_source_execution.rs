//! Test-only source checks above the SAME Worker storage/transition owners.
//! No source, file, Program, numeric factory, origin seal or router is created.
use super::*;
use crate::replicated_decoder::program::schedule_access::Kind;
use pllm_core::streamed_matrix::{MatrixGeometry, MatrixLayout, READ_BUFFER_BYTES};

fn reader_geometry(
    reader: PrivateScheduleRef<'_>,
    registry: &SourceRegistry<'_>,
) -> Result<(), Error> {
    let mut public = 0;
    for id in 0..reader.len() {
        let node = reader.node(id)?;
        match node.kind() {
            Kind::PublicInput => {
                if id != public {
                    return Err(Error::Binding);
                }
                let (key, digest) = node.public_input()?;
                registry.check_descriptor(id, key, *digest, node.lanes(), node.bounds())?;
                public += 1;
            }
            Kind::PublicProduct => {
                let (input, weight, rows, inner, columns, transposed) = node.public_product()?;
                let (key, digest) = reader.node(*weight)?.public_input()?;
                let (_, expected_key, source, _) =
                    registry.records().nth(*weight).ok_or(Error::Binding)?;
                if expected_key != key
                    || source.digest != *digest
                    || source.shape.as_slice()
                        != if *transposed {
                            [*inner, *columns]
                        } else {
                            [*columns, *inner]
                        }
                    || reader.node(*input)?.kind() == Kind::PublicInput
                {
                    return Err(Error::Binding);
                }
                let geometry = MatrixGeometry::new(
                    *rows,
                    *inner,
                    *columns,
                    if *transposed {
                        MatrixLayout::InputOutput
                    } else {
                        MatrixLayout::OutputInput
                    },
                )
                .map_err(|_| Error::Bounds)?;
                if geometry.input_elements() != reader.node(*input)?.lanes()
                    || geometry.output_elements() != node.lanes()
                {
                    return Err(Error::Shape);
                }
            }
            Kind::Gather => {
                for coordinate in node.lane_refs()? {
                    let (owner, index) = coordinate?;
                    let input = reader.node(owner)?;
                    if input.kind() == Kind::PublicInput {
                        return Err(Error::Binding);
                    }
                    if index >= input.lanes() {
                        return Err(Error::Shape);
                    }
                }
            }
            Kind::TensorLoop => {
                let input = *node.loop_input()?;
                let owner = match input {
                    LoopInput::Slice(owner) | LoopInput::TokenOffsets { owner, .. } => owner,
                };
                if reader.node(owner)?.kind() == Kind::PublicInput {
                    return Err(Error::Binding);
                }
                if let LoopInput::TokenOffsets { owner, index } = input {
                    if index >= reader.node(owner)?.lanes() {
                        return Err(Error::Shape);
                    }
                }
            }
            Kind::Input | Kind::PublicValues | Kind::Affine | Kind::Product | Kind::Tensor => {}
            // No per-word public gather, bulk/trie plan loan, iota, checked guard
            // or source-backed proof scope is silently coerced into this path.
            _ => return Err(Error::Binding),
        }
    }
    if public != registry.records().count() {
        return Err(Error::Binding);
    }
    Ok(())
}

pub(super) fn preflight(
    control: &[Worker; 3],
    candidate: &[Worker; 3],
    region: &FrozenRegion<'_>,
    registry: &SourceRegistry<'_>,
) -> Result<usize, Error> {
    // Exact borrowed metadata owners, NOT equal maps/digests or a source issued
    // for some foreign frozen recipe. Immutable files remain inside each consumed
    // original Worker; no replacement constructor participates in qualification.
    if !std::ptr::eq(region.borrowed_source_records(), registry.source_records())
        || !std::ptr::eq(region.borrowed_source_norms(), registry.source_norms())
        || region.source_binding() != registry.binding()
    {
        return Err(Error::Binding);
    }
    reader_geometry(candidate[0].program.schedule_access(), registry)?;
    for (id, _, source, _) in registry.records() {
        let (_, node) = candidate[0].program.input_at(id);
        match &node.code {
            Code::PublicInput {
                row_l1,
                row_squared_norm,
                row_l1_squared_sum,
                ..
            } if *row_l1 == source.row_l1
                && *row_squared_norm == source.row_squared_norm
                && *row_l1_squared_sum == source.row_l1_squared_sum => {}
            _ => return Err(Error::Binding),
        }
    }
    let mut bytes = READ_BUFFER_BYTES;
    for worker in control.iter().chain(candidate) {
        let public = worker.public.as_ref().ok_or(Error::Binding)?;
        public.check(&worker.program)?;
        public.check_registry(registry)?;
        bytes = add(bytes, public.resident_metadata_bytes()?)?;
    }
    // File byte length is disk backing, never a resident coefficient allocation.
    // Matrix scan workspace and fixed typed call/result slots coexist with the
    // already quoted output/physical storage and registry/source metadata.
    for control in [
        size_of::<SourceRegistry<'_>>(),
        size_of::<Option<&SourceRegistry<'_>>>(),
        2 * size_of::<MatrixGeometry>(),
        size_of::<Result<MatrixGeometry, pllm_core::streamed_matrix::Error>>(),
        2 * size_of::<[&[u64]; 2]>(),
        size_of::<[&mut [u64]; 2]>(),
        size_of::<Result<Secret, Error>>(),
        crate::replicated_decoder::source_registry::binding_workspace_bytes()?,
    ] {
        bytes = add(bytes, control)?;
    }
    Ok(bytes)
}
