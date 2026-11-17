//! Exact rows and their ownership link share ONE existing allocation. Bounded
//! const dispatch is intentional: no capacity rounding, DST casts or allocator.
use super::{Error, ExactBuffer};

pub(crate) trait Atom: Copy + Send + Sync + 'static {}
impl<T: Copy + Send + Sync + 'static> Atom for T {}
mod sealed { pub trait Sealed {} }
pub(crate) trait Link<T: Atom, P: Atom, Q: Atom>: sealed::Sealed + Send + Sync {
    fn rows(&self) -> &[T];
    fn copied(&self) -> usize;
    fn push(&mut self, row: T) -> Result<(), Error>;
    fn identity(&self) -> P;
    fn next(&self) -> &Chain<T, P, Q>;
    fn next_mut(&mut self) -> &mut Chain<T, P, Q>;
    fn proofs(&self) -> &[Q];
    fn proofs_slot(&mut self) -> &mut Box<[Q]>;
    fn allocation_bytes(&self) -> usize;
    fn allocation_layout(&self) -> std::alloc::Layout;
    #[cfg(test)] fn rows_mut(&mut self) -> &mut [T];
    #[cfg(test)] fn copied_mut(&mut self) -> &mut usize;
    #[cfg(test)] fn identity_mut(&mut self) -> &mut P;
}
pub(crate) struct Chain<T: Atom, P: Atom, Q: Atom> { head: Option<Box<dyn Link<T, P, Q>>> }
impl<T: Atom, P: Atom, Q: Atom> Chain<T, P, Q> {
    pub(crate) const fn empty() -> Self { Self { head: None } }
    // Preserve the owned Box's object bound; the REFERENCE still borrows self.
    // Eliding the object bound here narrows it to the borrow and loses that fact.
    pub(crate) fn first(&self) -> Option<&(dyn Link<T, P, Q> + 'static)> { self.head.as_deref() }
    pub(crate) fn prepend(&mut self, mut link: Box<dyn Link<T, P, Q>>) {
        // No callback/failure between these ownership writes. A fresh pending
        // link has an empty next; callers check readiness BEFORE commit.
        link.next_mut().head = self.head.take();
        self.head = Some(link);
    }
    pub(crate) fn find(&self, id: P) -> Option<&dyn Link<T, P, Q>> where P: Eq {
        let mut current = self.first();
        while let Some(link) = current {
            if link.identity() == id { return Some(link); }
            current = link.next().first();
        }
        None
    }
    #[cfg(test)]
    pub(crate) fn find_mut(&mut self, id: P) -> Option<&mut (dyn Link<T, P, Q> + 'static)> where P: Eq {
        let mut current = self.head.as_deref_mut();
        while let Some(link) = current {
            if link.identity() == id { return Some(link); }
            current = link.next_mut().head.as_deref_mut();
        }
        None
    }
    #[cfg(test)]
    pub(crate) fn remove(&mut self, id: P) -> Option<Box<dyn Link<T, P, Q>>> where P: Eq {
        let mut slot = &mut self.head;
        loop {
            if slot.as_ref()?.identity() == id {
                let mut removed = slot.take().unwrap();
                *slot = removed.next_mut().head.take();
                return Some(removed);
            }
            slot = &mut slot.as_mut()?.next_mut().head;
        }
    }
}
impl<T: Atom, P: Atom, Q: Atom> Drop for Chain<T, P, Q> {
    fn drop(&mut self) {
        // Also used while constructing a partial clone. Every detached record
        // has an empty next before its destructor/field glue is entered.
        while let Some(mut link) = self.head.take() {
            self.head = link.next_mut().head.take();
            drop(link);
        }
    }
}
impl<T: Atom, P: Atom, Q: Atom> Clone for Chain<T, P, Q> {
    fn clone(&self) -> Self {
        let mut result = Self::empty();
        let mut current = self.first();
        while let Some(source) = current {
            let seed = source.rows()[0];
            let link = allocate(source.rows().len(), seed, source.identity()).unwrap_or_else(|error| {
                match error {
                    Error::Allocation => std::alloc::handle_alloc_error(source.allocation_layout()),
                    _ => panic!("invalid linked clone extent"),
                }
            });
            result.prepend(link); // install BEFORE copying/proof allocation/unwind
            let destination = result.head.as_deref_mut().unwrap();
            for row in &source.rows()[..source.copied()] { destination.push(*row).unwrap(); }
            #[cfg(test)] tests::clone_seam();
            let mut proofs = ExactBuffer::new(source.proofs().len()).unwrap_or_else(|error| {
                match error {
                    Error::Allocation => std::alloc::handle_alloc_error(
                        std::alloc::Layout::array::<Q>(source.proofs().len()).expect("existing proof layout")),
                    _ => panic!("invalid linked proof extent"),
                }
            });
            for proof in source.proofs() { assert!(proofs.push(*proof).is_ok()); }
            *destination.proofs_slot() = proofs.finish().expect("complete proof clone");
            // No clone of source.next: chain order is not logical row identity.
            current = source.next().first();
        }
        result
    }
}

pub(crate) struct Record<T: Atom, P: Atom, Q: Atom, const N: usize> {
    rows: [T; N],
    copied: usize,
    identity: P,
    next: Chain<T, P, Q>,
    proofs: Box<[Q]>,
}
#[cfg(test)]
impl<T: Atom, P: Atom, Q: Atom, const N: usize> Drop for Record<T, P, Q, N> {
    fn drop(&mut self) { tests::record_drop(); }
}
impl<T: Atom, P: Atom, Q: Atom, const N: usize> sealed::Sealed for [Record<T, P, Q, N>; 1] {}
impl<T: Atom, P: Atom, Q: Atom, const N: usize> Link<T, P, Q> for [Record<T, P, Q, N>; 1] {
    fn rows(&self) -> &[T] { &self[0].rows }
    fn copied(&self) -> usize { self[0].copied }
    fn push(&mut self, row: T) -> Result<(), Error> {
        let record = &mut self[0];
        let slot = record.rows.get_mut(record.copied).ok_or(Error::Incomplete)?;
        *slot = row;
        record.copied += 1;
        Ok(())
    }
    fn identity(&self) -> P { self[0].identity }
    fn next(&self) -> &Chain<T, P, Q> { &self[0].next }
    fn next_mut(&mut self) -> &mut Chain<T, P, Q> { &mut self[0].next }
    fn proofs(&self) -> &[Q] { &self[0].proofs }
    fn proofs_slot(&mut self) -> &mut Box<[Q]> { &mut self[0].proofs }
    fn allocation_bytes(&self) -> usize { std::mem::size_of::<Self>() }
    fn allocation_layout(&self) -> std::alloc::Layout { std::alloc::Layout::new::<Self>() }
    #[cfg(test)] fn rows_mut(&mut self) -> &mut [T] { &mut self[0].rows }
    #[cfg(test)] fn copied_mut(&mut self) -> &mut usize { &mut self[0].copied }
    #[cfg(test)] fn identity_mut(&mut self) -> &mut P { &mut self[0].identity }
}
fn allocate_n<T: Atom, P: Atom, Q: Atom, const N: usize>(seed: T, identity: P) -> Result<Box<dyn Link<T, P, Q>>, Error> {
    let mut buffer = ExactBuffer::new(1)?;
    // Copy seed: no user Default/Clone/callback after allocation. The only
    // errors below are structurally impossible at this known single extent.
    let record = Record { rows: [seed; N], copied: 0, identity,
        next: Chain::empty(), proofs: Box::default() };
    assert!(buffer.push(record).is_ok());
    let slice = buffer.finish().expect("one initialized record");
    let array: Box<[Record<T, P, Q, N>; 1]> = match slice.try_into() {
        Ok(array) => array, Err(_) => unreachable!("one record extent"),
    };
    Ok(array) // standard unsizing, no allocation or copy
}
#[derive(Clone, Copy)]
pub(crate) struct Layouts { pub(crate) backing: usize, pub(crate) staging: usize }
fn layouts_n<T: Atom, P: Atom, Q: Atom, const N: usize>() -> Result<Layouts, Error> {
    // Mandatory const evaluation: no new runtime census helper/branch work in
    // checker/hash passes. Dispatch and the returned two-scalar ABI are unchanged.
    Record::<T, P, Q, N>::LAYOUTS
}
impl<T: Atom, P: Atom, Q: Atom, const N: usize> Record<T, P, Q, N> {
    // Acquisition, push, finish, conversion. Each includes its own source
    // bindings/formals/returns, not guessed machine frames or stdlib internals.
    // Exposed crate-privately ONLY as const metadata for independent owner tests.
    pub(crate) const STAGING_PHASES: Result<[usize; 4], Error> = 'census: {
        use std::mem::{size_of, ManuallyDrop, MaybeUninit};
        use std::ptr::NonNull;
        let common: &[usize] = &[
            size_of::<T>(), size_of::<T>(), size_of::<T>(),
            size_of::<P>(), size_of::<P>(),
            size_of::<usize>(), size_of::<usize>(), size_of::<usize>(),
            size_of::<ExactBuffer<Self>>(), // allocate_n's caller buffer slot
            size_of::<Result<Box<dyn Link<T, P, Q>>, Error>>(), // allocate_n return
            size_of::<Result<Box<dyn Link<T, P, Q>>, Error>>(), // dispatch return
        ];
        let acquisition: &[usize] = &[
            size_of::<usize>(), // ExactBuffer::new count
            size_of::<std::alloc::Layout>(), // new's layout binding
            size_of::<std::alloc::Layout>(), // allocate_nonzero's owned formal
            size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
            size_of::<Result<std::alloc::Layout, Error>>(),
            size_of::<std::alloc::LayoutError>(), // map_err argument
            size_of::<NonNull<MaybeUninit<Self>>>(), // pointer binding
            size_of::<*mut MaybeUninit<Self>>(), // raw binding after cast
            size_of::<*mut u8>(), // allocate_nonzero return, before cast
            size_of::<Option<NonNull<MaybeUninit<Self>>>>(),
            size_of::<Result<NonNull<MaybeUninit<Self>>, Error>>(),
            size_of::<*mut [MaybeUninit<Self>]>(), // from_raw's argument
            size_of::<Box<[MaybeUninit<Self>]>>(), // ownership construction
            size_of::<ExactBuffer<Self>>(), // constructed Self before Ok
            size_of::<Result<ExactBuffer<Self>, Error>>(), // new return
            size_of::<bool>(), // positive-layout branch
            size_of::<Error>(), // exclusive allocation/error control
        ];
        let push: &[usize] = &[
            size_of::<Self>(), // named seed record
            size_of::<Self>(), // push's owned value formal
            size_of::<Result<(), Self>>(),
            size_of::<&mut ExactBuffer<Self>>(),
            size_of::<Option<&mut MaybeUninit<Self>>>(),
            size_of::<&mut MaybeUninit<Self>>(), // slot binding
            size_of::<&Result<(), Self>>(), // is_ok's borrowed argument
            size_of::<bool>(), // assert control
        ];
        let finish: &[usize] = &[
            size_of::<ExactBuffer<Self>>(), // finish's consumed self formal
            size_of::<ManuallyDrop<ExactBuffer<Self>>>(), // this binding
            size_of::<Box<[MaybeUninit<Self>]>>(), // slots binding
            size_of::<&Box<[MaybeUninit<Self>]>>(), // borrow before ptr::read
            size_of::<*const Box<[MaybeUninit<Self>]>>(), // ptr::read argument
            size_of::<Box<[Self]>>(), // assume_init's owned return
            size_of::<Result<Box<[Self]>, Error>>(), // finish return
            size_of::<usize>(), size_of::<usize>(), // initialized/len controls
            size_of::<bool>(), size_of::<Error>(), // exclusive incomplete control
        ];
        let conversion: &[usize] = &[
            size_of::<Result<Box<[Self]>, Error>>(), // finish return
            size_of::<Box<[Self]>>(), // named slice
            size_of::<Result<Box<[Self; 1]>, Box<[Self]>>>(),
            size_of::<Box<[Self; 1]>>(), // match Ok(array) binding
            size_of::<Box<[Self; 1]>>(), // distinct outer array binding
            size_of::<Box<dyn Link<T, P, Q>>>(), // unsized construction temporary
        ];
        let phases = [acquisition, push, finish, conversion];
        let mut totals = [0usize; 4];
        let mut p = 0;
        while p < phases.len() {
            let mut part = 0;
            while part < 2 {
                let rows = if part == 0 { common } else { phases[p] };
                let mut i = 0;
                while i < rows.len() {
                    totals[p] = match totals[p].checked_add(rows[i]) {
                        Some(n) => n, None => break 'census Err(Error::Overflow),
                    };
                    i += 1;
                }
                part += 1;
            }
            p += 1;
        }
        Ok(totals)
    };
    const LAYOUTS: Result<Layouts, Error> = match Self::STAGING_PHASES {
        Err(error) => Err(error),
        Ok(phases) => {
            let mut peak = 0;
            let mut p = 0;
            while p < phases.len() {
                if phases[p] > peak { peak = phases[p]; }
                p += 1;
            }
            Ok(Layouts { backing: std::mem::size_of::<[Self; 1]>(), staging: peak })
        }
    };
}
macro_rules! dispatch_block {
    ($n:expr, $base:expr, $f:ident $args:tt; $($offset:literal),*) => {
        match $n { $(n if n == $base + $offset => $f::<T, P, Q, { $base + $offset }> $args,)*
            _ => Err(Error::Overflow) }
    };
}
macro_rules! dispatch {
    ($n:expr, $f:ident $args:tt; $($base:literal),*) => {
        match $n { $(n if n > $base && n <= $base + 16 =>
            dispatch_block!(n, $base, $f $args; 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),)*
            _ => Err(Error::Overflow) }
    };
}
pub(crate) fn allocate<T: Atom, P: Atom, Q: Atom>(n: usize, seed: T, identity: P) -> Result<Box<dyn Link<T, P, Q>>, Error> {
    dispatch!(n, allocate_n(seed, identity); 0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368)
}
pub(crate) fn layouts<T: Atom, P: Atom, Q: Atom>(n: usize) -> Result<Layouts, Error> {
    dispatch!(n, layouts_n(); 0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368)
}

#[cfg(test)] mod tests;
