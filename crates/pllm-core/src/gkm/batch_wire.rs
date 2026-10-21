//! Consuming source-stage aggregates over the original scalar window owners.
use super::*;
use crate::gkm::batch::{BatchPayload, WorkerBatchPayload, AGGREGATE_HEADER_BYTES};
use crate::gkm::{BatchAdmission, BatchDescriptor};

const BATCH_MAGIC: &[u8; 8] = b"PLGKMB01";

pub struct ClientBatchWireWindow {
    client: ClientWireWindow,
    payload: Option<BatchPayload>,
    ordinal: usize,
    pending: bool,
}

pub struct WorkerBatchWireWindow {
    worker: WorkerWireWindow,
    ordinal: usize,
}

pub(super) fn owner_control_bytes() -> Result<usize, Error> {
    let client = size_of::<ClientBatchWireWindow>()
        .checked_sub(size_of::<ClientWireWindow>())
        .ok_or(Error::Limit)?;
    let worker = size_of::<WorkerBatchWireWindow>()
        .checked_sub(size_of::<WorkerWireWindow>())
        .ok_or(Error::Limit)?;
    Ok(client
        .max(
            worker
                .checked_add(size_of::<WorkerBatchPayload>())
                .ok_or(Error::Limit)?,
        )
        .max(size_of::<BatchPayload>()))
}

impl Client {
    /// Explicit selected execution; sharing comes from the trusted, consuming
    /// admission token. Original scalar/seeded constructors remain unchanged.
    pub fn begin_batch_wire(
        &self,
        context: [u8; 32],
        scalar_plan: &[Step],
        admission: BatchAdmission,
    ) -> Result<ClientBatchWireWindow, Error> {
        if let Err(error) =
            admission.validate_binding(&self.publication, scalar_plan, admission.profile)
        {
            // Streamed attempt is irreversibly spent without a lease, payload or
            // seed allocation. This does not create a retry/rewind capability.
            let _ = self.publication.reserve_window();
            return Err(error);
        }
        let (window, keys) = match admission.profile {
            SharingProfile::FullShares => (self.begin(context, scalar_plan)?, None),
            SharingProfile::Aes128WindowV1 => {
                let (window, keys) = self.seeded_wire_parts(context, scalar_plan)?;
                (window, Some(keys))
            }
        };
        window
            ._lease
            .attach_batch(admission, &window.plan, window.profile)?;
        Ok(ClientBatchWireWindow {
            client: window.into_wire_inner(keys)?,
            payload: None,
            ordinal: 0,
            pending: false,
        })
    }
}

impl WorkerWire {
    fn reject_batch_start(&mut self) {
        if let Some(source) = &self.source {
            let _ = source.budget.publication.reserve_window();
        }
        self.abort();
    }

    /// Bound counterpart to begin_bound, with all application/start/admission
    /// binding checks before any scalar plan, seed, stream or lease allocation.
    pub fn begin_batch_bound(
        &mut self,
        authenticated_start: &[u8],
        expected_context: [u8; 32],
        expected_plan: &[Step],
        admission: BatchAdmission,
    ) -> Result<WorkerBatchWireWindow, Error> {
        let checked = (|| {
            let frame = self.validate_start_envelope(authenticated_start)?;
            self.validate_start_schedule(&frame)?;
            if frame.binding.context != expected_context {
                return Err(Error::Binding);
            }
            let source = self.source.as_ref().ok_or(Error::Retired)?;
            admission.validate_binding(
                &source.budget.publication,
                expected_plan,
                frame.binding.profile,
            )
        })();
        if let Err(error) = checked {
            self.reject_batch_start();
            return Err(error);
        }
        self.begin_batch(authenticated_start, admission)
    }

    pub fn begin_batch(
        &mut self,
        authenticated_start: &[u8],
        admission: BatchAdmission,
    ) -> Result<WorkerBatchWireWindow, Error> {
        let checked = (|| {
            let frame = self.validate_start_envelope(authenticated_start)?;
            self.validate_start_schedule(&frame)?;
            let source = self.source.as_ref().ok_or(Error::Retired)?;
            admission.validate_wire_binding(
                &source.budget.publication,
                &frame.body[..frame.binding.total as usize * 33],
                frame.binding.total,
                frame.binding.profile,
            )
        })();
        if let Err(error) = checked {
            self.reject_batch_start();
            return Err(error);
        }
        let result = self.begin_inner_admitted(authenticated_start, Some(admission));
        match result {
            Ok(worker) => Ok(WorkerBatchWireWindow { worker, ordinal: 0 }),
            Err(error) => {
                self.abort();
                Err(error)
            }
        }
    }
}

impl ClientWireWindow {
    /// Consumes a fresh, unstarted scalar/seeded transport owner. The approved
    /// batch selector must come from trusted local SDK configuration.
    pub fn into_batch(mut self, admission: BatchAdmission) -> Result<ClientBatchWireWindow, Error> {
        let Some(ClientState::Ready(window)) = &self.state else {
            return Err(Error::Retired);
        };
        if self.started || window.completed != 0 {
            return Err(Error::Schedule);
        }
        window
            ._lease
            .attach_batch(admission, &window.plan, window.profile)?;
        // The selected lease now reserves the complete R+S, replacing the scalar
        // transport allowance. Native scalar row workspace remains native-priced.
        self._permit = WirePermit::acquire(0, window.publication.capacity)?;
        Ok(ClientBatchWireWindow {
            client: self,
            payload: None,
            ordinal: 0,
            pending: false,
        })
    }
}

impl WorkerWireWindow {
    pub fn into_batch(mut self, admission: BatchAdmission) -> Result<WorkerBatchWireWindow, Error> {
        if self.completed != 0 || self.run.status.load(Ordering::Acquire) != ACTIVE {
            return Err(Error::Retired);
        }
        self.run
            .lease
            .attach_batch(admission, &self.plan, self.binding.profile)?;
        self._permit = WirePermit::acquire(0, self.source.budget.publication.capacity)?;
        Ok(WorkerBatchWireWindow {
            worker: self,
            ordinal: 0,
        })
    }
}

fn admission(lease: &Arc<Lease>) -> Result<Arc<BatchAdmission>, Error> {
    lease.ensure_open()?;
    lease.batch_admission()?.ok_or(Error::Binding)
}

fn descriptor(
    admission: &BatchAdmission,
    ordinal: usize,
    claim: usize,
) -> Result<BatchDescriptor, Error> {
    let descriptor = *admission
        .plan
        .descriptors()
        .get(ordinal)
        .ok_or(Error::Schedule)?;
    if descriptor.first_claim as usize != claim {
        return Err(Error::Schedule);
    }
    Ok(descriptor)
}

fn row_length(
    publication: &Publication,
    descriptor: BatchDescriptor,
    kind: u8,
    worker: Worker,
    profile: SharingProfile,
) -> Result<usize, Error> {
    let (rows, columns) = shape(publication, descriptor.step)?;
    let (input, output) = match descriptor.step {
        Step::Lookup => (rows.div_ceil(8), columns),
        Step::Linear(_) => (columns, rows),
    };
    let [request, reply] = crate::gkm::batch::row_payload_lengths(descriptor.step, input, output)?;
    match kind {
        REQUEST if worker == Worker::One && profile == SharingProfile::Aes128WindowV1 => Ok(0),
        REQUEST => Ok(request),
        REPLY => Ok(reply),
        _ => Err(Error::Binding),
    }
}

fn encoded_length(descriptor: BatchDescriptor, row_length: usize) -> Result<usize, Error> {
    crate::gkm::batch::aggregate_length(descriptor.rows, row_length)
}

// Explicit wire bindings and the pre-reserved frame permit travel together.
#[allow(clippy::too_many_arguments)]
fn encode_header(
    kind: u8,
    binding: Binding,
    dimensions: (usize, usize),
    admission: &BatchAdmission,
    ordinal: usize,
    descriptor: BatchDescriptor,
    length: usize,
    lease: &Arc<Lease>,
    permit: FramePermit,
) -> Result<WireBytes, Error> {
    let mut out = WireBytes::allocate_leased(length, lease, permit, true)?;
    out.extend(BATCH_MAGIC);
    out.extend(&[
        kind,
        if binding.profile == SharingProfile::FullShares {
            0
        } else {
            1
        },
    ]);
    out.extend(&Sha256::digest(NUMERIC_PROFILE));
    out.extend(&binding.publication);
    out.extend(&binding.session);
    out.extend(&binding.sequence.to_le_bytes());
    out.extend(&binding.context);
    out.extend(&binding.plan);
    out.extend(&binding.total.to_le_bytes());
    out.extend(&admission.plan.digest());
    out.extend(
        &u32::try_from(ordinal)
            .map_err(|_| Error::Limit)?
            .to_le_bytes(),
    );
    out.extend(&descriptor.first_claim.to_le_bytes());
    out.extend(&descriptor.rows.to_le_bytes());
    out.extend(&step_bytes(binding.step));
    out.extend(&binding.artifact);
    out.extend(&[if binding.worker == Worker::Zero { 0 } else { 1 }]);
    out.extend(&request_id(&binding));
    out.extend(
        &u32::try_from(dimensions.0)
            .map_err(|_| Error::Limit)?
            .to_le_bytes(),
    );
    out.extend(
        &u32::try_from(dimensions.1)
            .map_err(|_| Error::Limit)?
            .to_le_bytes(),
    );
    out.extend(
        &u64::try_from(length)
            .map_err(|_| Error::Limit)?
            .to_le_bytes(),
    );
    if out.len() != AGGREGATE_HEADER_BYTES {
        return Err(Error::Binding);
    }
    Ok(out)
}

fn decode<'a>(
    bytes: &'a [u8],
    kind: u8,
    expected: Binding,
    publication: &Publication,
    admission: &BatchAdmission,
    ordinal: usize,
    descriptor: BatchDescriptor,
) -> Result<&'a [u8], Error> {
    let row_length = row_length(
        publication,
        descriptor,
        kind,
        expected.worker,
        expected.profile,
    )?;
    let length = encoded_length(descriptor, row_length)?;
    if bytes.len() != length || length > admission.resources().max_aggregate_bytes {
        return Err(Error::Shape);
    }
    let mut cursor = Cursor::new(bytes);
    if cursor.take(8)? != BATCH_MAGIC || cursor.byte()? != kind {
        return Err(Error::Binding);
    }
    let profile = match cursor.byte()? {
        0 => SharingProfile::FullShares,
        1 => SharingProfile::Aes128WindowV1,
        _ => return Err(Error::Binding),
    };
    if cursor.take(32)? != Sha256::digest(NUMERIC_PROFILE).as_slice() {
        return Err(Error::Binding);
    }
    let publication_pin = cursor.array()?;
    let session = cursor.array()?;
    let sequence = cursor.u64()?;
    let context = cursor.array()?;
    let plan = cursor.array()?;
    let total = cursor.u32()?;
    if cursor.array::<32>()? != admission.plan.digest()
        || cursor.u32()? as usize != ordinal
        || cursor.u32()? != descriptor.first_claim
        || cursor.u32()? != descriptor.rows
    {
        return Err(Error::Binding);
    }
    let step = read_step(&mut cursor)?;
    let artifact = cursor.array()?;
    let worker = match cursor.byte()? {
        0 => Worker::Zero,
        1 => Worker::One,
        _ => return Err(Error::Binding),
    };
    let binding = Binding {
        publication: publication_pin,
        session,
        sequence,
        context,
        plan,
        total,
        ordinal: descriptor.first_claim,
        step,
        artifact,
        worker,
        profile,
    };
    if binding != expected
        || cursor.array::<32>()? != request_id(&binding)
        || (cursor.u32()? as usize, cursor.u32()? as usize) != shape(publication, descriptor.step)?
        || cursor.u64()? != u64::try_from(length).map_err(|_| Error::Limit)?
        || cursor.position() != AGGREGATE_HEADER_BYTES
    {
        return Err(Error::Binding);
    }
    let body = &bytes[AGGREGATE_HEADER_BYTES..];
    for _ in 0..descriptor.rows {
        if cursor.u64()? != row_length as u64 {
            return Err(Error::Shape);
        }
        let row = cursor.take(row_length)?;
        if kind == REQUEST && descriptor.step == Step::Lookup && row_length != 0 {
            let used = publication.row_hashes.len() % 8;
            if used != 0 && row[row.len() - 1] >> used != 0 {
                return Err(Error::Shape);
            }
        }
    }
    cursor.finish()?;
    Ok(body)
}

fn next_row<'a>(cursor: &mut Cursor<'a>) -> Result<&'a [u8], Error> {
    let length = usize::try_from(cursor.u64()?).map_err(|_| Error::Limit)?;
    cursor.take(length)
}

impl ClientBatchWireWindow {
    pub fn request_id(&self) -> [u8; 32] {
        self.client.request_id()
    }

    pub fn start_frames(&mut self) -> Result<[WireBytes; 2], Error> {
        let result = self.client.start_frames();
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn linear(&mut self, input: &[i64]) -> Result<[WireBytes; 2], Error> {
        let result = self.issue(Some(input), None);
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn lookup(&mut self, tokens: &[usize]) -> Result<[WireBytes; 2], Error> {
        let result = self.issue(None, Some(tokens));
        if result.is_err() {
            self.abort();
        }
        result
    }

    fn issue(
        &mut self,
        input: Option<&[i64]>,
        tokens: Option<&[usize]>,
    ) -> Result<[WireBytes; 2], Error> {
        let mut window = self.client.take_ready()?;
        if self.pending {
            return Err(Error::Schedule);
        }
        let admission = admission(&window._lease)?;
        let descriptor = descriptor(&admission, self.ordinal, window.completed)?;
        // Validate the complete operation before any flat owner allocation or
        // entropy consumption. take_ready already consumes malformed attempts.
        match descriptor.step {
            Step::Linear(id) => {
                let values = input.ok_or(Error::Schedule)?;
                if tokens.is_some() {
                    return Err(Error::Shape);
                }
                let matrix = &window.publication.matrices[window.publication.matrix_index(&id)?];
                crate::gkm::numeric::validate_inputs(
                    values,
                    matrix.columns,
                    &matrix.input_bounds,
                    descriptor.rows as usize,
                )?;
            }
            Step::Lookup => {
                let values = tokens.ok_or(Error::Schedule)?;
                if input.is_some() || values.len() != descriptor.rows as usize {
                    return Err(Error::Shape);
                }
                if values
                    .iter()
                    .any(|&token| token >= window.publication.row_hashes.len())
                {
                    return Err(Error::Bound);
                }
            }
        }
        // Previous tentative batch borrows end at this mutable transition. Drop
        // that owner before allocating the next stage; there is no owner queue.
        self.payload = None;
        let mut payload =
            BatchPayload::allocate(&window.publication, Arc::clone(&admission), self.ordinal)?;
        match descriptor.step {
            Step::Linear(_) => {
                let input = input.ok_or(Error::Schedule)?;
                if tokens.is_some() || input.len() != payload.input.len() {
                    return Err(Error::Shape);
                }
                payload.input.copy_from_slice(input);
                let columns = input.len() / descriptor.rows as usize;
                for (row, input) in input.chunks_exact(columns).enumerate() {
                    let PendingLinear {
                        input: _,
                        shares,
                        slots: _,
                        stage: _,
                        window: next,
                    } = window.linear_with(input, &mut OsEntropy)?;
                    window = next;
                    for (role, words) in shares.iter().enumerate() {
                        let stride = words.len() * 8;
                        let begin = row * stride;
                        for (out, word) in payload.shares[role][begin..begin + stride]
                            .chunks_exact_mut(8)
                            .zip(words.iter())
                        {
                            out.copy_from_slice(&word.to_le_bytes());
                        }
                    }
                }
            }
            Step::Lookup => {
                let tokens = tokens.ok_or(Error::Schedule)?;
                if input.is_some() || tokens.len() != payload.tokens.len() {
                    return Err(Error::Shape);
                }
                payload.tokens.copy_from_slice(tokens);
                for (row, &token) in tokens.iter().enumerate() {
                    let PendingLookup {
                        token: _,
                        selectors,
                        slots: _,
                        window: next,
                    } = window.lookup_with(token, &mut OsEntropy)?;
                    window = next;
                    for (role, bytes) in selectors.iter().enumerate() {
                        let begin = row * bytes.len();
                        payload.shares[role][begin..begin + bytes.len()].copy_from_slice(bytes);
                    }
                }
            }
        }
        let lengths = [Worker::Zero, Worker::One].map(|role| {
            row_length(
                &window.publication,
                descriptor,
                REQUEST,
                role,
                window.profile,
            )
            .and_then(|n| encoded_length(descriptor, n))
        });
        let [left, right] =
            FramePermit::aggregate_pair(&window._lease, [lengths[0]?, lengths[1]?])?;
        let encode = |role: Worker, permit: FramePermit| -> Result<WireBytes, Error> {
            let binding = window.binding(role)?;
            let stride = row_length(
                &window.publication,
                descriptor,
                REQUEST,
                role,
                window.profile,
            )?;
            let mut out = encode_header(
                REQUEST,
                binding,
                shape(&window.publication, descriptor.step)?,
                &admission,
                self.ordinal,
                descriptor,
                encoded_length(descriptor, stride)?,
                &window._lease,
                permit,
            )?;
            let shares = &payload.shares[usize::from(role == Worker::One)];
            for row in 0..descriptor.rows as usize {
                out.extend(&(stride as u64).to_le_bytes());
                out.extend(&shares[row * stride..(row + 1) * stride]);
            }
            Ok(out)
        };
        let frames = [encode(Worker::Zero, left)?, encode(Worker::One, right)?];
        self.payload = Some(payload);
        self.pending = true;
        self.client.state = Some(ClientState::Ready(window));
        Ok(frames)
    }

    pub fn accept_replies(&mut self, zero: &[u8], one: &[u8]) -> Result<WireTentative<'_>, Error> {
        if let Err(error) = self.accept_inner(zero, one) {
            self.abort();
            return Err(error);
        }
        let payload = self.payload.as_ref().ok_or(Error::Retired)?;
        Ok(WireTentative {
            values: &payload.values,
            signed: &payload.signed,
        })
    }

    fn accept_inner(&mut self, zero: &[u8], one: &[u8]) -> Result<(), Error> {
        self.accept_inner_with(zero, one, &mut OsEntropy)
    }

    fn accept_inner_with(
        &mut self,
        zero: &[u8],
        one: &[u8],
        rng: &mut impl Entropy,
    ) -> Result<(), Error> {
        let mut window = self.client.take_ready()?;
        if !self.pending {
            return Err(Error::Schedule);
        }
        let admission = admission(&window._lease)?;
        let descriptor = descriptor(&admission, self.ordinal, window.completed)?;
        for (role, bytes) in [(Worker::Zero, zero), (Worker::One, one)] {
            decode(
                bytes,
                REPLY,
                window.binding(role)?,
                &window.publication,
                &admission,
                self.ordinal,
                descriptor,
            )?;
        }
        let freeze = |bytes: &[u8]| -> Result<WireScratch<u8>, Error> {
            let mut frozen = WireScratch::filled(&window._lease, bytes.len(), 0u8)?;
            frozen.copy_from_slice(bytes);
            Ok(frozen)
        };
        // Entire role packets are immutable and owned before the first alpha.
        let zero = freeze(zero)?;
        let one = freeze(one)?;
        let mut zero = Cursor::new(&zero[AGGREGATE_HEADER_BYTES..]);
        let mut one = Cursor::new(&one[AGGREGATE_HEADER_BYTES..]);
        let payload = self.payload.as_mut().ok_or(Error::Retired)?;
        for row in 0..descriptor.rows as usize {
            let a = next_row(&mut zero)?;
            let b = next_row(&mut one)?;
            match descriptor.step {
                Step::Linear(id) => {
                    let stage = window.publication.matrix_index(&id)?;
                    let matrix = &window.publication.matrices[stage];
                    let columns = matrix.columns;
                    let pending = PendingLinear {
                        stage,
                        input: Zeroizing::new(payload::copy(
                            &payload.input[row * columns..(row + 1) * columns],
                        )?),
                        shares: [Zeroizing::new(Vec::new()), Zeroizing::new(Vec::new())],
                        slots: [ResponseSlot::new(), ResponseSlot::new()],
                        window,
                    };
                    let decode = |role, bytes: &[u8]| -> Result<LinearResponse, Error> {
                        let share = pending.share(role);
                        let mut words = WireScratch::filled(share.lease, share.rows, 0u64)?;
                        for (word, bytes) in words.iter_mut().zip(bytes.chunks_exact(8)) {
                            *word = u64::from_le_bytes(bytes.try_into().map_err(|_| Error::Shape)?);
                        }
                        LinearResponse::from_untrusted(&share, share.binding(), &words)
                    };
                    let left = decode(Worker::Zero, a)?;
                    let right = decode(Worker::One, b)?;
                    acknowledge_seeded_reply(&pending.window);
                    window = pending
                        .freeze_first(Ok(left))?
                        .freeze_second_with(Ok(right), rng)?;
                }
                Step::Lookup => {
                    let pending = PendingLookup {
                        token: Zeroizing::new(payload.tokens[row]),
                        selectors: [Zeroizing::new(Vec::new()), Zeroizing::new(Vec::new())],
                        slots: [ResponseSlot::new(), ResponseSlot::new()],
                        window,
                    };
                    let decode = |role, bytes: &[u8]| {
                        let share = pending.share(role);
                        LookupResponse::from_untrusted(&share, share.binding(), bytes)
                    };
                    let left = decode(Worker::Zero, a)?;
                    let right = decode(Worker::One, b)?;
                    acknowledge_seeded_reply(&pending.window);
                    window = pending.freeze_first(Ok(left))?.freeze_second(Ok(right))?;
                }
            }
            let width = window.tentative.len();
            payload.values[row * width..(row + 1) * width].copy_from_slice(&window.tentative);
            if !window.signed.is_empty() {
                payload.signed[row * width..(row + 1) * width].copy_from_slice(&window.signed);
            }
        }
        zero.finish()?;
        one.finish()?;
        self.pending = false;
        self.ordinal += 1;
        self.client.state = Some(ClientState::Ready(window));
        Ok(())
    }

    pub fn finish(&mut self) -> Result<AcceptedWindow, Error> {
        let result = (|| {
            if self.pending {
                return Err(Error::Incomplete);
            }
            let mut accepted = self.client.finish()?;
            let mut payload = self.payload.take().ok_or(Error::Incomplete)?;
            accepted.values = Zeroizing::new(std::mem::take(&mut *payload.values));
            accepted.signed = Zeroizing::new(std::mem::take(&mut *payload.signed));
            Ok(accepted)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub fn abort(&mut self) {
        self.client.abort();
        self.payload = None;
        self.pending = false;
    }
}

impl WorkerBatchWireWindow {
    pub fn request_id(&self) -> [u8; 32] {
        self.worker.request_id()
    }
    pub fn sharing_profile(&self) -> SharingProfile {
        self.worker.sharing_profile()
    }

    pub fn execute(&mut self, bytes: &[u8]) -> Result<WireBytes, Error> {
        let result = self.execute_inner(bytes);
        if result.is_err() {
            self.worker.run.fail();
        }
        result
    }

    fn execute_inner(&mut self, bytes: &[u8]) -> Result<WireBytes, Error> {
        if self.worker.run.status.load(Ordering::Acquire) != ACTIVE {
            return Err(Error::Retired);
        }
        let admission = admission(&self.worker.run.lease)?;
        let descriptor = descriptor(&admission, self.ordinal, self.worker.completed)?;
        let publication = &self.worker.source.budget.publication;
        let mut expected = self.worker.binding;
        expected.ordinal = self.worker.completed as u32;
        expected.step = descriptor.step;
        expected.artifact = artifact(publication, descriptor.step)?;
        let body = decode(
            bytes,
            REQUEST,
            expected,
            publication,
            &admission,
            self.ordinal,
            descriptor,
        )?;
        let dimensions = shape(publication, descriptor.step)?;
        let stride = row_length(
            publication,
            descriptor,
            REPLY,
            expected.worker,
            expected.profile,
        )?;
        let length = encoded_length(descriptor, stride)?;
        let permit = FramePermit::aggregate(&self.worker.run.lease, length)?;
        let mut out = encode_header(
            REPLY,
            expected,
            dimensions,
            &admission,
            self.ordinal,
            descriptor,
            length,
            &self.worker.run.lease,
            permit,
        )?;
        let batch = descriptor.rows as usize;
        let (input_width, output_width) = match descriptor.step {
            Step::Linear(id) => {
                let matrix = &publication.matrices[publication.matrix_index(&id)?];
                (matrix.columns, matrix.rows)
            }
            Step::Lookup => (publication.row_hashes.len().div_ceil(8), publication.width),
        };
        let mut workspace =
            WorkerBatchPayload::allocate(&admission, &descriptor, input_width, output_width)?;
        let seeded =
            expected.worker == Worker::One && expected.profile == SharingProfile::Aes128WindowV1;
        if seeded {
            // decode already authenticated every zero-length share record.
            let seed = self.worker.seeded.as_mut().ok_or(Error::Binding)?;
            match descriptor.step {
                Step::Linear(_) => seed.fill_linear_rows(
                    expected,
                    &self.worker.run.lease,
                    batch,
                    &mut workspace.linear_input,
                )?,
                Step::Lookup => seed.fill_lookup_rows(
                    expected,
                    &self.worker.run.lease,
                    batch,
                    &mut workspace.selectors,
                )?,
            }
        } else {
            let mut cursor = Cursor::new(body);
            for token in 0..batch {
                let request = next_row(&mut cursor)?;
                match descriptor.step {
                    Step::Linear(_) => {
                        for (value, bytes) in workspace.linear_input
                            [token * input_width..(token + 1) * input_width]
                            .iter_mut()
                            .zip(request.chunks_exact(8))
                        {
                            *value =
                                u64::from_le_bytes(bytes.try_into().map_err(|_| Error::Shape)?);
                        }
                    }
                    Step::Lookup => workspace.selectors
                        [token * input_width..(token + 1) * input_width]
                        .copy_from_slice(request),
                }
            }
            cursor.finish()?;
        }
        // Exactly one shared dense primitive dispatch for the complete operation.
        match descriptor.step {
            Step::Linear(id) => {
                let index = publication.matrix_index(&id)?;
                self.worker.source.matrices[index].multiply_shares_into(
                    &workspace.linear_input,
                    batch,
                    &mut workspace.linear_output,
                )?;
                for row in workspace.linear_output.chunks_exact(output_width) {
                    out.extend(&(stride as u64).to_le_bytes());
                    for value in row {
                        out.extend(&value.to_le_bytes());
                    }
                }
            }
            Step::Lookup => {
                self.worker.source.table.xor_selectors_into(
                    &workspace.selectors,
                    batch,
                    &mut workspace.bytes,
                )?;
                for row in workspace.bytes.chunks_exact(stride) {
                    out.extend(&(stride as u64).to_le_bytes());
                    out.extend(row);
                }
            }
        }
        if out.as_bytes().len() != length {
            return Err(Error::Shape);
        }
        self.worker.run.lease.ensure_open()?;
        if self.worker.run.status.load(Ordering::Acquire) != ACTIVE {
            return Err(Error::Retired);
        }
        // No watermark moves until the entire aggregate has been serialized.
        if seeded {
            self.worker
                .seeded
                .as_mut()
                .ok_or(Error::Binding)?
                .commit_rows(batch)?;
        }
        self.worker.completed = self
            .worker
            .completed
            .checked_add(batch)
            .ok_or(Error::Limit)?;
        self.ordinal += 1;
        Ok(out)
    }

    pub fn finish(&mut self) -> Result<WorkerWireReceipt, Error> {
        self.worker.finish()
    }
    pub fn abort(&mut self) {
        self.worker.abort();
    }
}

#[cfg(test)]
#[path = "batch_wire_tests.rs"]
mod tests;
