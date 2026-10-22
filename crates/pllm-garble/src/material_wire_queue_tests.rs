//! Actual queue source plus actual private packet fixture, without a public
//! spare-capacity constructor or a second implementation of the queue.
use super::{tests::with_spare, Budget, Error, MaterialPacket};

// Only these public descriptor fields and callback methods are read by Queues.
// Full compiler descriptor/transcript/private-feedback tests remain in compiler.
// Visibility matches the included module's signatures; this entire unit-test
// module is private. No packet constructor or fixture is exposed outside it.
#[derive(Clone, Copy)]
pub struct MaterialRequest {
    role: u8,
    packet_bytes_bound: usize,
}
pub trait DealerTransport {
    fn check_active(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn deliver_batch(
        &mut self,
        role: u8,
        records: &[batch::MaterialRecord<'_>],
    ) -> Result<(), Error>;
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}

// Compile the real module, not copied logic. Test-only parent imports select the
// same neutral Packet/Budget here and in compiler's normal unit-test context.
#[allow(dead_code)] // This fixture targets capacity paths, not every queue API.
#[path = "../../pllm-compiler/src/fss_decoder/transport_batch.rs"]
mod batch;

#[derive(Default)]
struct Sink {
    // Public recipient and original packet addresses only; no packet copies.
    calls: Vec<(u8, Vec<usize>)>,
}
impl DealerTransport for Sink {
    fn deliver_batch(
        &mut self,
        role: u8,
        records: &[batch::MaterialRecord<'_>],
    ) -> Result<(), Error> {
        let pointers = records
            .iter()
            .map(|record| {
                assert_eq!(record.request().role, role);
                assert!(record.packet().len() <= record.request().packet_bytes_bound);
                record.packet().as_ptr() as usize
            })
            .collect();
        self.calls.push((role, pointers));
        Ok(())
    }
}
fn requests(bound: usize) -> [MaterialRequest; 2] {
    [0, 1].map(|role| MaterialRequest {
        role,
        packet_bytes_bound: bound,
    })
}

#[test]
fn spare_backing_length_preflight_then_capacity_refusal_keeps_prior_queue() {
    for refused_role in 0..2 {
        for invalid_length in [false, true] {
            let destination = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
            let source = Budget::new(crate::shared_allocation::MAX_ALLOCATION_BYTES).unwrap();
            let mut queues = batch::Queues::new(&destination).unwrap();
            let mut sink = Sink::default();
            let prior = [with_spare(&source, 0), with_spare(&source, 0)];
            let len = prior[0].as_bytes().len();
            let pointers = prior
                .each_ref()
                .map(|packet| packet.as_bytes().as_ptr() as usize);
            queues.push_pair(requests(len), prior).unwrap();
            assert_eq!(source.used_bytes(), 0);
            assert_eq!(destination.used_bytes(), 2 * len);
            assert!(!batch::Queues::singleton(len).unwrap());
            assert!(!queues.needs_flush(len).unwrap()); // Wire-length preflight fits.
            let before_body =
                batch::MATERIAL_BATCH_ENVELOPE_BYTES_BOUND + batch::record_bytes(len).unwrap();
            let capacity = batch::MATERIAL_BATCH_BODY_BYTES_BOUND
                - before_body
                - batch::record_bytes(0).unwrap()
                + 1;
            assert!(capacity > len);
            let mut packets = [with_spare(&source, 0), with_spare(&source, 0)];
            // Replacing this test owner drops its original charge normally.
            packets[refused_role] = with_spare(&source, capacity - len);
            assert!(
                packets[refused_role].capacity_bytes() > packets[refused_role].as_bytes().len()
            );
            assert!(packets.iter().all(|packet| packet.as_bytes().len() == len));
            assert_eq!(source.used_bytes(), capacity + len);
            let mut request = requests(len);
            if invalid_length {
                request[refused_role].packet_bytes_bound -= 1;
            }
            assert_eq!(
                queues.push_pair(request, packets),
                Err(if invalid_length {
                    Error::Context // Length guard precedes capacity guard for this role.
                } else {
                    Error::Resource // Both lengths fit; actual backing does not.
                })
            );
            assert_eq!(source.used_bytes(), 0); // Both rejected originals retired.
            assert_eq!(destination.used_bytes(), 2 * len); // Prior owners unchanged.
            assert!(!queues.needs_flush(len).unwrap());
            assert!(!queues.full());
            assert!(sink.calls.is_empty()); // Failure did not flush prior owners.
            queues.flush(&mut sink).unwrap();
            assert_eq!(
                sink.calls,
                vec![(0, vec![pointers[0]]), (1, vec![pointers[1]])]
            );
            assert!(queues.complete(1));
            assert_eq!(destination.used_bytes(), 0);
            assert_eq!(source.used_bytes(), 0);
        }
    }
}

#[test]
fn admitted_spare_backing_drives_next_length_flush_and_retires_original_pointers() {
    let resources = batch::BatchResources::new().unwrap();
    assert_eq!(resources.body_bytes_bound(), 1 << 20);
    assert_eq!(resources.records_bound(), 64);
    let destination = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
    let source = Budget::new(crate::shared_allocation::MAX_ALLOCATION_BYTES).unwrap();
    let mut queues = batch::Queues::new(&destination).unwrap();
    let mut sink = Sink::default();
    let len = with_spare(&source, 0).as_bytes().len();
    assert_eq!(source.used_bytes(), 0);
    let capacity = batch::MATERIAL_BATCH_BODY_BYTES_BOUND
        - batch::MATERIAL_BATCH_ENVELOPE_BYTES_BOUND
        - batch::record_bytes(0).unwrap()
        - batch::record_bytes(len).unwrap()
        + 1;
    let packets = [
        with_spare(&source, capacity - len),
        with_spare(&source, capacity - len),
    ];
    let pointers = packets
        .each_ref()
        .map(|packet| packet.as_bytes().as_ptr() as usize);
    assert!(packets
        .iter()
        .all(|packet| packet.capacity_bytes() == capacity && packet.as_bytes().len() == len));
    assert!(!queues.needs_flush(len).unwrap());
    queues.push_pair(requests(len), packets).unwrap();
    assert_eq!(source.used_bytes(), 0);
    assert_eq!(destination.used_bytes(), 2 * capacity);
    assert!(!queues.full()); // Not a count/exact-body-boundary flush.
    assert!(
        batch::MATERIAL_BATCH_ENVELOPE_BYTES_BOUND + 2 * batch::record_bytes(len).unwrap()
            < batch::MATERIAL_BATCH_BODY_BYTES_BOUND
    ); // Length-only accounting would fit.
    assert!(queues.needs_flush(len).unwrap()); // Charged spare backing forces flush.
    assert!(sink.calls.is_empty());
    queues.flush(&mut sink).unwrap();
    assert_eq!(
        sink.calls,
        vec![(0, vec![pointers[0]]), (1, vec![pointers[1]])]
    );
    assert!(queues.complete(1));
    assert_eq!(destination.used_bytes(), 0);
    assert!(!queues.needs_flush(len).unwrap());
    let tail = [with_spare(&source, 0), with_spare(&source, 0)];
    queues.push_pair(requests(len), tail).unwrap();
    assert_eq!(source.used_bytes(), 0);
    assert_eq!(destination.used_bytes(), 2 * len);
    queues.flush(&mut sink).unwrap();
    assert!(queues.complete(2));
    assert_eq!(
        sink.calls.iter().map(|call| call.0).collect::<Vec<_>>(),
        [0, 1, 0, 1]
    );
    assert_eq!(destination.used_bytes(), 0);
    assert_eq!(source.used_bytes(), 0);
}
