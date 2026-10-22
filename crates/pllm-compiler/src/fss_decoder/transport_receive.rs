//! Public opt-in and native owner quote for local frozen-group provision.
use super::{add, Error, MaterialRequest};
use pllm_garble::fusefss_backend::composition::{provider::ReplyWindow, Plan};

pub(super) type Window<'a> = ReplyWindow<'a, MaterialRequest>;
pub const MATERIAL_RECEIVE_PYTHON_BYTES: usize =
    8 * super::MATERIAL_WINDOW_BYTES + 8192 * super::MATERIAL_WINDOW_RECORDS + (64 << 10);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterialReceive {
    #[default]
    Scalar,
    FrozenWindow,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReceiveResources {
    native: usize,
}
impl ReceiveResources {
    pub(super) fn new() -> Result<Self, Error> {
        Ok(Self {
            native: Window::reservation_bytes()?,
        })
    }
    pub fn native_reservation_bytes(&self) -> usize {
        self.native
    }
    pub fn python_overlap_bytes(&self) -> usize {
        MATERIAL_RECEIVE_PYTHON_BYTES
    }
}
pub(super) fn prepare(
    window: &mut Window<'_>,
    plan: &Plan<'_>,
    count: usize,
    packet_bound: usize,
) -> Result<(), Error> {
    debug_assert!(window.requests().is_empty());
    let mut bytes = 0;
    // Prefix of this frozen group only, using each original public packet bound.
    for index in 0..count {
        let pc = window.pc(index)?;
        let bound = plan.step_transport_resources(pc)?.packet_bytes_bound;
        if bound > packet_bound {
            return Err(Error::Resource);
        }
        let next = add(bytes, bound)?;
        if next > super::MATERIAL_WINDOW_BYTES {
            break;
        }
        let mut request = window.template();
        request.pc = pc;
        request.packet_bytes_bound = bound;
        window.push(request, bound)?;
        bytes = next;
    }
    if window.requests().is_empty() {
        return Err(Error::Resource);
    }
    Ok(())
}
