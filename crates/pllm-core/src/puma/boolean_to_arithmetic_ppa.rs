//! Source B2AByPPA: mask in Boolean space, reveal only x+r to role zero,
//! then rotate additive output components. Reuses the existing carry owner.

use super::{Plan, Share, State as ParentState, Status};
use crate::puma::multiply::conversion;
use crate::puma::prss::{blocks, fill, SourceParty};
use crate::puma::*;
use sha2::{Digest, Sha256};

// All fixed metadata stays inside the runtime's checked 4096-byte reserve.
#[allow(clippy::large_enum_variant)]
pub(crate) enum State {
    Split {
        plan: Plan,
        random: Words,
        first: Words,
        input: Share,
    },
    Carry {
        plan: Plan,
        random: Words,
        work: conversion::Work,
    },
    Reveal {
        plan: Plan,
        first: Words,
        second: Words,
    },
    Rotate {
        plan: Plan,
        first: Words,
    },
}

impl State {
    pub(super) fn status(&self) -> Status {
        match self {
            Self::Reveal { .. } => Status::AwaitingOpen,
            Self::Rotate { .. } => Status::AwaitingReshare,
            _ => Status::AwaitingRotate,
        }
    }
}

fn carry(plan: Plan) -> Result<conversion::Plan, Error> {
    conversion::Plan::nested_adder(
        plan.operation,
        plan.setup,
        plan.pair_start + 2 * blocks(plan.lanes())?,
        plan.limits,
    )
}

fn operation(plan: Plan, phase: MessagePhase) -> Result<Operation, Error> {
    let tag = match phase {
        MessagePhase::BooleanPpaMask => 0_u8,
        MessagePhase::BooleanPpaReveal => 1,
        MessagePhase::BooleanPpaRotate => 2,
        _ => return Err(Error::Phase),
    };
    let mut operation = plan.operation;
    let mut hash = Sha256::new();
    hash.update(b"pllm.puma.b2a-ppa-message.v1");
    hash.update(plan.digest());
    hash.update([tag]);
    operation.consumer = Some(hash.finalize().into());
    Ok(operation)
}

fn destination(phase: MessagePhase, from: Party) -> Result<Party, Error> {
    match phase {
        MessagePhase::BooleanPpaReveal if from == Party::P2 => Ok(Party::P0),
        MessagePhase::BooleanPpaMask | MessagePhase::BooleanPpaRotate => Ok(from.next().next()),
        _ => Err(Error::Role),
    }
}

fn frame(plan: Plan, phase: MessagePhase, from: Party, words: Words) -> Result<Message, Error> {
    if words.len() != plan.lanes() {
        return Err(Error::Shape);
    }
    Ok(Message {
        header: Header {
            binding: operation(plan, phase)?.digest(),
            from,
            to: destination(phase, from)?,
            phase,
        },
        words,
    })
}

fn accept(
    source: &SourceParty,
    plan: Plan,
    phase: MessagePhase,
    mut message: Message,
) -> Result<Words, Error> {
    let from = if phase == MessagePhase::BooleanPpaReveal {
        Party::P2
    } else {
        source.party().next()
    };
    if destination(phase, from)? != source.party() {
        return Err(Error::Role);
    }
    message.validate(operation(plan, phase)?, from, source.party(), phase)?;
    message.words.charge_to(&source.payload_usage())?;
    Ok(message.words)
}

pub(super) fn import(
    plan: Plan,
    header: Header,
    words: &[u64],
    limits: Limits,
) -> Result<Message, Error> {
    if header.phase == MessagePhase::BooleanAnd {
        let carry = carry(plan)?;
        for round in 0..=6 {
            let gate = carry.gate(round)?;
            if gate.digest() == header.binding {
                return gate.message_from_received(header, words, limits);
            }
        }
        return Err(Error::Binding);
    }
    if destination(header.phase, header.from)? != header.to {
        return Err(Error::Role);
    }
    Message::from_received(header, words, operation(plan, header.phase)?, limits)
}

pub(super) fn issue(
    source: &mut SourceParty,
    plan: Plan,
    mut input: Share,
) -> Result<(ParentState, [Option<Message>; 2]), Error> {
    let n = plan.lanes();
    let q = blocks(n)?;
    source.cursors.aes_work = source.work(4 * q)?;
    source.cursors.pair_blocks = plan.pair_start + 2 * q;
    let usage = source.payload_usage();
    let seeds = source.seeds.as_ref().ok_or(Error::Retired)?;
    let adjacent = seeds.next.as_ref().ok_or(Error::Phase)?;
    let mut random = usage.zeros(n)?;
    let mut next_random = usage.zeros(n)?;
    let mut first = usage.zeros(n)?;
    let mut next_zero = usage.zeros(n)?;
    fill(&seeds.own, plan.pair_start, &mut random)?;
    fill(adjacent, plan.pair_start, &mut next_random)?;
    fill(&seeds.own, plan.pair_start + q, &mut first)?;
    fill(adjacent, plan.pair_start + q, &mut next_zero)?;
    for i in 0..n {
        first[i] ^= next_zero[i];
        if source.party() == Party::P1 {
            first[i] ^= random[i].wrapping_add(next_random[i]);
        }
    }
    input.bits = 64; // public widening, preserving original backing words
    let message = frame(
        plan,
        MessagePhase::BooleanPpaMask,
        source.party(),
        usage.copy(&first, n)?,
    )?;
    Ok((
        ParentState::Ppa(State::Split {
            plan,
            random,
            first,
            input,
        }),
        [Some(message), None],
    ))
}

pub(super) fn advance(
    source: &mut SourceParty,
    state: State,
    message: Message,
) -> Result<(ParentState, [Option<Message>; 2]), Error> {
    let usage = source.payload_usage();
    match state {
        State::Split {
            plan,
            random,
            first,
            input,
        } => {
            let second = accept(source, plan, MessagePhase::BooleanPpaMask, message)?;
            let rhs = Share {
                value: ReplicatedShare {
                    operation: plan.operation,
                    party: source.party(),
                    first,
                    second,
                },
                bits: 64,
            };
            let (work, outgoing) = conversion::start(source, carry(plan)?, input, rhs)?;
            Ok((
                ParentState::Ppa(State::Carry { plan, random, work }),
                [Some(outgoing), None],
            ))
        }
        State::Carry { plan, random, work } => {
            let (next, outgoing) = conversion::advance(source, work, message)?;
            match next {
                conversion::State::Circuit(work) => Ok((
                    ParentState::Ppa(State::Carry { plan, random, work }),
                    [outgoing, None],
                )),
                conversion::State::Complete(output) => {
                    if outgoing.is_some() || source.cursors.pair_blocks != plan.pair_end {
                        return Err(Error::Binding);
                    }
                    after_carry(source, plan, random, output)
                }
                _ => Err(Error::Phase),
            }
        }
        State::Reveal {
            plan,
            mut first,
            second,
        } => {
            let received = accept(source, plan, MessagePhase::BooleanPpaReveal, message)?;
            for i in 0..plan.lanes() {
                first[i] ^= second[i] ^ received[i];
            }
            // Only x+r is available here; P0 lacks one independent summand of r.
            let reply = frame(
                plan,
                MessagePhase::BooleanPpaRotate,
                source.party(),
                usage.copy(&first, plan.lanes())?,
            )?;
            Ok((
                ParentState::Ppa(State::Rotate { plan, first }),
                [Some(reply), None],
            ))
        }
        State::Rotate { plan, first } => {
            let second = accept(source, plan, MessagePhase::BooleanPpaRotate, message)?;
            Ok((
                ParentState::Complete(ReplicatedShare {
                    operation: plan.operation,
                    party: source.party(),
                    first,
                    second,
                }),
                [None, None],
            ))
        }
    }
}

fn after_carry(
    source: &SourceParty,
    plan: Plan,
    mut random: Words,
    output: Share,
) -> Result<(ParentState, [Option<Message>; 2]), Error> {
    if source.party() == Party::P0 {
        drop(random);
        return Ok((
            ParentState::Ppa(State::Reveal {
                plan,
                first: output.value.first,
                second: output.value.second,
            }),
            [None, None],
        ));
    }
    let reveal = if source.party() == Party::P2 {
        Some(frame(
            plan,
            MessagePhase::BooleanPpaReveal,
            source.party(),
            output.value.first,
        )?)
    } else {
        None
    };
    for value in &mut *random {
        *value = value.wrapping_neg();
    }
    let rotate = frame(
        plan,
        MessagePhase::BooleanPpaRotate,
        source.party(),
        source.payload_usage().copy(&random, plan.lanes())?,
    )?;
    Ok((
        ParentState::Ppa(State::Rotate {
            plan,
            first: random,
        }),
        [reveal, Some(rotate)],
    ))
}
