//! Canonical bounded codecs for one party's masked-wire material and messages.
//!
//! Authentication and admission/replay ledgers belong to the receiving session.
//! Material packets are secret: send each only to its assigned authenticated
//! party. Neither packet contains the client's input or output masks. Decoding
//! is preflighted against the same combined allocation bound as local issuance.

use super::*;
use crate::compact_dpf::{BitReader, BitSource, DecodeError};

const MATERIAL: &[u8; 8] = b"PLLMMK01";
const SEEDED_MATERIAL: &[u8; 8] = b"PLLMMK02";
const INPUT: &[u8; 8] = b"PLLMMI01";
const OUTPUT: &[u8; 8] = b"PLLMMO01";
pub const PACKET_HEADER: usize = 76;

impl From<DecodeError> for Error {
    fn from(_: DecodeError) -> Self {
        Self::Context
    }
}

fn read_shift(input: &mut impl BitSource, n: u8, shift: u8, gap: bool) -> Result<Shift, Error> {
    let dpf = Key::decode(shift, input)?;
    let width = usize::from(if gap { n } else { n - shift });
    let mut constants = Zeroizing::new([0; 3]);
    constants[0] = input.random(width, RandomField::FirstParty)? as u64;
    if gap {
        constants[1] = input.random(width, RandomField::FirstParty)? as u64;
    }
    constants[2] = input.random(width, RandomField::FirstParty)? as u64;
    Ok(Shift {
        dpf,
        n,
        shift,
        gap,
        constants,
    })
}

fn read_select(input: &mut impl BitSource, bits: u8, rows: usize) -> Result<Select, Error> {
    let mut coefficients = Zeroizing::new(Vec::with_capacity(rows));
    for _ in 0..rows {
        coefficients.push([
            input.random(bits.into(), RandomField::FirstParty)? as u64,
            input.random(bits.into(), RandomField::FirstParty)? as u64,
        ]);
    }
    Ok(Select { bits, coefficients })
}

fn read_lane(input: &mut impl BitSource, layout: Layout) -> Result<Lane, Error> {
    let gate = read_shift(input, N, 12, true)?;
    let up = read_shift(input, N, 12, true)?;
    let (n, shift) = layout.reduce_bits();
    let reduce = read_shift(input, n, shift, false)?;
    let signs = Signs {
        key: Key::decode(layout.sign_bits() - 1, input)?,
        bits: layout.sign_bits(),
        scale: if layout == Layout::Fused { 18 } else { 0 },
        constants: Zeroizing::new([
            input.random(1, RandomField::FirstParty)? as u64,
            input.random(1, RandomField::FirstParty)? as u64,
        ]),
    };
    let clip = read_select(input, 10, 4)?;
    let relu = read_select(input, N, 2)?;
    let lookup = Lookup {
        dpf: Key::decode(10, input)?,
        constants: Zeroizing::new([
            input.random(N.into(), RandomField::FirstParty)? as u64,
            input.random(N.into(), RandomField::FirstParty)? as u64,
        ]),
    };
    let product = Multiply {
        coefficients: Zeroizing::new([
            input.random(N.into(), RandomField::FirstParty)? as u64,
            input.random(N.into(), RandomField::FirstParty)? as u64,
            input.random(N.into(), RandomField::FirstParty)? as u64,
        ]),
    };
    let output = read_shift(input, N, 12, true)?;
    Ok(Lane {
        gate,
        up,
        reduce,
        signs,
        clip,
        relu,
        lookup,
        product,
        output,
    })
}

/// Exact two-party delivery lengths; expanded key allocation remains charged.
pub fn material_packet_bytes(
    lanes: usize,
    layout: Layout,
    material: MaterialEncoding,
) -> Result<[usize; 2], Error> {
    let resource = resources_with_layout(lanes, layout)?;
    match material {
        MaterialEncoding::Full => Ok([PACKET_HEADER + resource.party_key_bytes; 2]),
        MaterialEncoding::Seeded => Ok([
            PACKET_HEADER + 32 + seeded_codec::bits(lanes, layout, 0)?.div_ceil(8),
            PACKET_HEADER + 32 + seeded_codec::bits(lanes, layout, 1)?.div_ceil(8),
        ]),
    }
}

/// Encoded secret payload only; includes fresh seeds but excludes packet headers.
pub fn material_payload_bits(
    lanes: usize,
    layout: Layout,
    material: MaterialEncoding,
) -> Result<[usize; 2], Error> {
    let resource = resources_with_layout(lanes, layout)?;
    Ok(match material {
        MaterialEncoding::Full => [resource.party_key_bits; 2],
        MaterialEncoding::Seeded => [
            seeded_codec::bits(lanes, layout, 0)? + 256,
            seeded_codec::bits(lanes, layout, 1)? + 256,
        ],
    })
}

/// Consumes the issuer's party-local handle, retaining only its secret packet.
pub fn export_party(mut party: Party) -> Result<Zeroizing<Vec<u8>>, Error> {
    if !party.alive || party.phase.is_some() {
        return Err(Error::Consumed);
    }
    let resource = resources_with_layout(party.lanes, party.layout)?;
    let expected =
        material_packet_bytes(party.lanes, party.layout, party.material)?[usize::from(party.id)];
    let body = if party.material == MaterialEncoding::Seeded {
        let seed = party.seed.as_ref().ok_or(Error::Consumed)?;
        let mut writer = seeded_codec::Writer {
            body: Bits::new(),
            valid: true,
            tape: coins::Tape::new(seed, &party.binding, &party.nonce, party.id),
            party: party.id,
        };
        for lane in &party.keys {
            lane.encode(&mut writer);
        }
        if !writer.valid
            || writer.body.len != seeded_codec::bits(party.lanes, party.layout, party.id)?
        {
            return Err(Error::Resource);
        }
        writer.body
    } else {
        let mut body = Bits::new();
        for lane in &party.keys {
            lane.encode(&mut body);
        }
        if body.len != resource.party_key_bits {
            return Err(Error::Resource);
        }
        body
    };
    let mut bytes = Zeroizing::new(Vec::with_capacity(expected));
    bytes.extend(if party.material == MaterialEncoding::Seeded {
        SEEDED_MATERIAL
    } else {
        MATERIAL
    });
    bytes.extend(party.binding);
    bytes.extend(party.nonce);
    bytes.push(party.id);
    bytes.push(u8::from(party.layout == Layout::Fused));
    bytes.extend((party.lanes as u16).to_le_bytes());
    if let Some(seed) = &party.seed {
        bytes.extend_from_slice(&**seed);
    }
    bytes.extend_from_slice(&body.data);
    if bytes.len() != expected {
        return Err(Error::Resource);
    }
    party.cancel();
    Ok(bytes)
}

/// Imports only the expected role, layout and compiler/session/tensor context.
/// The caller must burn its monotonic session admission before calling this.
pub fn import_party(
    bytes: &[u8],
    context: Context,
    lanes: usize,
    layout: Layout,
    party: u8,
) -> Result<Party, Error> {
    import_party_with_material(bytes, context, lanes, layout, party, MaterialEncoding::Full)
}

pub fn import_party_with_material(
    bytes: &[u8],
    context: Context,
    lanes: usize,
    layout: Layout,
    party: u8,
    material: MaterialEncoding,
) -> Result<Party, Error> {
    let resource = resources_with_layout(lanes, layout)?;
    context
        .first_tensor_index
        .checked_add(lanes as u64)
        .ok_or(Error::Domain)?;
    let expected_binding = binding_with_material(context, lanes, layout, material);
    let sizes = material_packet_bytes(lanes, layout, material)?;
    if party > 1
        || bytes.len() != sizes[usize::from(party)]
        || &bytes[..8]
            != if material == MaterialEncoding::Seeded {
                SEEDED_MATERIAL
            } else {
                MATERIAL
            }
        || bytes[8..40] != expected_binding
        || bytes[72] != party
        || bytes[73] != u8::from(layout == Layout::Fused)
        || bytes[74..76] != (lanes as u16).to_le_bytes()
    {
        return Err(Error::Context);
    }
    let mut keys = Vec::with_capacity(lanes);
    let nonce = bytes[40..72].try_into().map_err(|_| Error::Context)?;
    if material == MaterialEncoding::Seeded {
        let seed = Zeroizing::new(
            <[u8; 32]>::try_from(&bytes[PACKET_HEADER..PACKET_HEADER + 32])
                .map_err(|_| Error::Context)?,
        );
        let mut reader = seeded_codec::Reader {
            body: BitReader::new(
                &bytes[PACKET_HEADER + 32..],
                seeded_codec::bits(lanes, layout, party)?,
            )?,
            tape: coins::Tape::new(&seed, &expected_binding, &nonce, party),
            party,
        };
        for _ in 0..lanes {
            keys.push(read_lane(&mut reader, layout)?);
        }
        reader.body.finish()?;
    } else {
        let mut reader = BitReader::new(&bytes[PACKET_HEADER..], resource.party_key_bits)?;
        for _ in 0..lanes {
            keys.push(read_lane(&mut reader, layout)?);
        }
        reader.finish()?;
    }
    Ok(Party {
        id: party,
        layout,
        material,
        seed: None,
        binding: expected_binding,
        nonce,
        lanes,
        keys,
        state: Zeroizing::new(vec![[0; 8]; lanes]),
        last: Zeroizing::new(Vec::new()),
        phase: None,
        alive: true,
        table: *Profile::new().table(),
    })
}

fn packet(
    magic: &[u8; 8],
    binding: &[u8; 32],
    nonce: &[u8; 32],
    lanes: usize,
    body: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(PACKET_HEADER + body.len());
    out.extend(magic);
    out.extend(binding);
    out.extend(nonce);
    out.extend((lanes as u32).to_le_bytes());
    out.extend(body);
    out
}

fn read_packet<'a>(
    bytes: &'a [u8],
    magic: &[u8; 8],
    binding: &[u8; 32],
    nonce: &[u8; 32],
    lanes: usize,
    columns: usize,
) -> Result<BitReader<'a>, Error> {
    if !(1..=MAX_LANES).contains(&lanes) || !(1..=2).contains(&columns) {
        return Err(Error::Resource);
    }
    let bits = lanes * columns * usize::from(N);
    if bytes.len() != PACKET_HEADER + bits.div_ceil(8)
        || &bytes[..8] != magic
        || bytes[8..40] != *binding
        || bytes[40..72] != *nonce
        || bytes[72..76] != (lanes as u32).to_le_bytes()
    {
        return Err(Error::Context);
    }
    Ok(BitReader::new(&bytes[PACKET_HEADER..], bits)?)
}

/// One-use client input masking; integer domain checks precede any output bytes.
pub fn encode_input(client: &mut Client, gate: &[i64], up: &[i64]) -> Result<Vec<u8>, Error> {
    let inputs = client.encode(gate, up)?;
    let mut body = Bits::new();
    for row in inputs.values.iter() {
        for &x in row {
            body.put(x.into(), N.into());
        }
    }
    Ok(packet(
        INPUT,
        &inputs.binding,
        &inputs.nonce,
        inputs.values.len(),
        &body.data,
    ))
}

/// Burns all party-local material if framing or session context is malformed.
pub fn start(party: &mut Party, bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let result = (|| {
        let mut reader = read_packet(bytes, INPUT, &party.binding, &party.nonce, party.lanes, 2)?;
        let mut values = Zeroizing::new(Vec::with_capacity(party.lanes));
        for _ in 0..party.lanes {
            values.push([reader.take(N.into())? as u64, reader.take(N.into())? as u64]);
        }
        reader.finish()?;
        party.start(&MaskedInputs {
            binding: party.binding,
            nonce: party.nonce,
            values,
        })
    })();
    if result.is_err() {
        party.cancel();
    }
    result
}

pub fn encode_output(output: MaskedOutput) -> Vec<u8> {
    let mut body = Bits::new();
    for &x in output.values.iter() {
        body.put(x.into(), N.into());
    }
    packet(
        OUTPUT,
        &output.binding,
        &output.nonce,
        output.values.len(),
        &body.data,
    )
}

/// Consumes client masks even on malformed or inconsistent party outputs.
pub fn decode_output(mut client: Client, left: &[u8], right: &[u8]) -> Result<Vec<i64>, Error> {
    if left != right {
        return Err(Error::Context);
    }
    let lanes = client.outputs.as_ref().ok_or(Error::Consumed)?.len();
    let mut reader = read_packet(left, OUTPUT, &client.binding, &client.nonce, lanes, 1)?;
    let mut values = Zeroizing::new(Vec::with_capacity(lanes));
    for _ in 0..lanes {
        values.push(reader.take(N.into())? as u64);
    }
    reader.finish()?;
    client.decode(&MaskedOutput {
        binding: client.binding,
        nonce: client.nonce,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 129,
        }
    }

    #[test]
    fn party_local_packets_execute_both_complete_layouts_and_exact_frame_costs() {
        for layout in [Layout::Unfused, Layout::Fused] {
            for material in [MaterialEncoding::Full, MaterialEncoding::Seeded] {
                let gate = [-(48 << 24), -1, 0, 1, 48 << 24];
                let up = [128 << 24, -4097, -1, 1, -(128 << 24)];
                let (mut client, parties) =
                    issue_reference_with_material(context(), gate.len(), layout, material).unwrap();
                let packets = parties.map(|p| export_party(p).unwrap());
                let resources = resources_with_layout(gate.len(), layout).unwrap();
                assert_eq!(
                    packets.each_ref().map(|p| p.len()),
                    material_packet_bytes(gate.len(), layout, material).unwrap()
                );
                let mut a = import_party_with_material(
                    &packets[0],
                    context(),
                    gate.len(),
                    layout,
                    0,
                    material,
                )
                .unwrap();
                let mut b = import_party_with_material(
                    &packets[1],
                    context(),
                    gate.len(),
                    layout,
                    1,
                    material,
                )
                .unwrap();
                let input = encode_input(&mut client, &gate, &up).unwrap();
                let (mut left, mut right) = (
                    start(&mut a, &input).unwrap(),
                    start(&mut b, &input).unwrap(),
                );
                let mut cost = left.len() + right.len();
                let mut result = None;
                for _ in 0..layout.rounds() {
                    match (a.advance(&right).unwrap(), b.advance(&left).unwrap()) {
                        (Step::Send(l), Step::Send(r)) => {
                            cost += l.len() + r.len();
                            left = l;
                            right = r;
                        }
                        (Step::Done(l), Step::Done(r)) => {
                            result = Some((encode_output(l), encode_output(r)));
                        }
                        _ => panic!("phase mismatch"),
                    }
                }
                assert_eq!(cost, resources.peer_frame_bytes);
                let (l, r) = result.unwrap();
                let output = decode_output(client, &l, &r).unwrap();
                let profile = Profile::new();
                for ((g, u), y) in gate.into_iter().zip(up).zip(output) {
                    assert_eq!(y, profile.evaluate(g, u).unwrap());
                }
                assert!(a.advance(&right).is_err());
                assert!(b.advance(&left).is_err());
            }
        }
    }

    #[test]
    fn seeded_roundtrip_preserves_every_expanded_key_bit_and_rejects_substitution() {
        for layout in [Layout::Unfused, Layout::Fused] {
            for lanes in [1, 3, 64, MAX_LANES] {
                let (_, parties) = issue_reference_with_material(
                    context(),
                    lanes,
                    layout,
                    MaterialEncoding::Seeded,
                )
                .unwrap();
                assert_ne!(parties[0].seed, parties[1].seed);
                for (id, party) in parties.into_iter().enumerate() {
                    let mut expanded = Bits::new();
                    for key in &party.keys {
                        key.encode(&mut expanded);
                    }
                    let packet = export_party(party).unwrap();
                    assert!(packet.len() < PACKET_HEADER + expanded.data.len());
                    let decoded = import_party_with_material(
                        &packet,
                        context(),
                        lanes,
                        layout,
                        id as u8,
                        MaterialEncoding::Seeded,
                    )
                    .unwrap();
                    let mut actual = Bits::new();
                    for key in &decoded.keys {
                        key.encode(&mut actual);
                    }
                    assert_eq!(*actual.data, *expanded.data);
                    assert!(decoded.seed.is_none());
                    assert!(import_party(&packet, context(), lanes, layout, id as u8).is_err());
                    assert!(import_party_with_material(
                        &packet,
                        context(),
                        lanes,
                        layout,
                        (1 - id) as u8,
                        MaterialEncoding::Seeded
                    )
                    .is_err());
                    let mut wrong = context();
                    wrong.first_tensor_index += 1;
                    assert!(import_party_with_material(
                        &packet,
                        wrong,
                        lanes,
                        layout,
                        id as u8,
                        MaterialEncoding::Seeded
                    )
                    .is_err());
                    for length in [0, 75, 107, packet.len() - 1] {
                        assert!(import_party_with_material(
                            &packet[..length],
                            context(),
                            lanes,
                            layout,
                            id as u8,
                            MaterialEncoding::Seeded
                        )
                        .is_err());
                    }
                    if seeded_codec::bits(lanes, layout, id as u8).unwrap() % 8 != 0 {
                        let mut padded = packet.clone();
                        *padded.last_mut().unwrap() |= 0x80;
                        assert!(import_party_with_material(
                            &padded,
                            context(),
                            lanes,
                            layout,
                            id as u8,
                            MaterialEncoding::Seeded
                        )
                        .is_err());
                    }
                }
            }
            let (_, [mut first, _]) =
                issue_reference_with_material(context(), 1, layout, MaterialEncoding::Seeded)
                    .unwrap();
            first.seed.as_mut().unwrap()[0] ^= 1;
            assert!(export_party(first).is_err());
            let (_, [full, _]) = issue_reference_with_layout(context(), 1, layout).unwrap();
            assert!(import_party_with_material(
                &export_party(full).unwrap(),
                context(),
                1,
                layout,
                0,
                MaterialEncoding::Seeded
            )
            .is_err());
        }
    }

    #[test]
    fn foreign_material_truncation_padding_and_malformed_inputs_reject() {
        let layout = Layout::Fused;
        let (mut client, [a, _]) = issue_reference_with_layout(context(), 1, layout).unwrap();
        let bytes = export_party(a).unwrap();
        let mut wrong = context();
        wrong.first_tensor_index += 1;
        assert!(import_party(&bytes, wrong, 1, layout, 0).is_err());
        assert!(import_party(&bytes, context(), 1, Layout::Unfused, 0).is_err());
        assert!(import_party(&bytes, context(), 1, layout, 1).is_err());
        assert!(import_party(&bytes, context(), MAX_LANES + 1, layout, 0).is_err());
        for length in [0, 1, 75, bytes.len() - 1] {
            assert!(import_party(&bytes[..length], context(), 1, layout, 0).is_err());
        }
        for index in [0, 8, 39, 72, 73, 74, 75] {
            let mut forged = bytes.clone();
            forged[index] ^= 0xff;
            assert!(import_party(&forged, context(), 1, layout, 0).is_err());
        }
        let mut padded = bytes.clone();
        *padded.last_mut().unwrap() |= 0x80;
        assert!(import_party(&padded, context(), 1, layout, 0).is_err());
        let mut party = import_party(&bytes, context(), 1, layout, 0).unwrap();
        let valid = encode_input(&mut client, &[0], &[0]).unwrap();
        let mut invalid = valid.clone();
        invalid[40] ^= 1;
        assert!(start(&mut party, &invalid).is_err());
        assert!(start(&mut party, &valid).is_err());
        assert!(party.keys.is_empty());
    }
}
