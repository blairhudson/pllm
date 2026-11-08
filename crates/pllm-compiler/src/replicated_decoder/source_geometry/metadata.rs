//! Inactive SOURCE inventory candidate, not funded admission or activation.
//! Entry and body guards are unpriced. No production caller enters this module.
//! Do not replace either guard with byte arithmetic, zero work or a test token.
use super::*;
use pllm_core::exact_buffer::{self, ExactBuffer};
use pllm_models::{checkpoint_requirements as checker, owned_backing};
use pllm_types::bounded_canonical::borrowed::NativeFrame;
use std::ops::Range;

// Declaration order is retirement order, including error/unwind ownership.
// SOURCEEVENT notation below: N=operation_slots, R=requirement_slots,
// F=attribute_frame_slots, U=unique records <= R, B=genuine Header::text_bytes(),
// T=copied name bytes <= B. Counts describe successful paths or visited failure
// prefixes, not prices. All candidate body paths remain unreachable/unfunded.
struct MetadataPreparation {
    permit: original_source::MetadataEntryPermit,
    geometry: GeometryOwner,
}

struct Publication {
    records: Records,
    preparation: MetadataPreparation,
}
type MetadataOwner = OwnedCharged<Publication>;

// One packed allocation, not one String per name. Construction validates the
// entire B-byte zero buffer once; queries only check range/UTF-8 boundaries.
struct PackedString(String);
struct Record {
    name: Range<usize>,
    shape: checker::Shape,
}
struct Records {
    entries: Box<[Record]>,
    names: PackedString,
}

// bounded_index's adapter targets its own Error and consumes rejected values.
// Use Core directly; this tiny local adapter adds no storage/allocator wrapper.
fn exact_error(_: exact_buffer::Error) -> Error { Error::Bounds }

impl Records {
    fn name(&self, index: usize) -> Option<&str> {
        let entry = self.entries.get(index)?;
        self.names.0.get(entry.name.start..entry.name.end)
    }

    // Uncharged private builder, called ONLY inside publication.try_build.
    // Publication owns this backing before its guard; no independent escape.
    fn copy(
        view: &checker::RequirementsView<'_, '_>,
        text_bound: usize,
        requirement_bound: usize,
    ) -> Result<Self, Error> {
        // U is known only after genuine check; U<=R is checked before allocation.
        // Retained publication charge STILL covers R descriptors + B bytes, not
        // merely U/T. No slack refund. layout_bytes(U) repeats layout validation;
        // Core new(U) also checks Layout::array, then owns initialized prefixes.
        let count = view.records().len();
        if count > requirement_bound { return Err(Error::Bounds); }
        exact_buffer::layout_bytes::<Record>(count).map_err(exact_error)?;
        let mut entries = ExactBuffer::new(count).map_err(exact_error)?;
        // Packed backing: one new(B), B literal-zero pushes, one finish. Push
        // refusal returns incoming u8: explicitly retire it before translation;
        // remaining initialized prefix/backing retires through Core Drop. Zero
        // length is supported by Core without entering the allocator.
        let mut bytes = ExactBuffer::new(text_bound).map_err(exact_error)?;
        for _ in 0..text_bound {
            if let Err(value) = bytes.push(0_u8) {
                drop(value);
                return Err(Error::Bounds);
            }
        }
        let bytes = bytes.finish().map_err(exact_error)?;
        // Box<[u8]>::into_vec transfers backing, len/capacity=B, no allocation.
        // String::from_utf8 consumes that Vec, scans FULL B bytes, and on success
        // retains its backing without copying. This validation is UNPRICED, not
        // waived because source names are valid &str. Error owns the Vec; retire
        // it explicitly before Binding, then entries retires. Result/Vec/String
        // descriptors, FromUtf8Error and move/drop overlap need control inventory.
        let mut names = String::from_utf8(bytes.into_vec()).map_err(|error| {
            drop(error.into_bytes());
            Error::Binding
        })?;
        names.clear(); // length 0, fixed B capacity retained; no backing release.
        for record in view.records() {
            let start = names.len();
            let end = plus(start, record.name.len())?;
            // Before EVERY push_str: checked end<=B AND actual capacity. No
            // growth path; one append per unique record, aggregate T bytes.
            if end > text_bound || end > names.capacity() { return Err(Error::Bounds); }
            names.push_str(record.name);
            // Checked range AND character boundaries before descriptor/publication.
            // get checks endpoints, not another whole-buffer UTF-8 scan.
            if names.get(start..end).is_none() { return Err(Error::Binding); }
            if let Err(value) = entries.push(Record { name: start..end, shape: record.shape }) {
                drop(value);
                return Err(Error::Bounds);
            }
        }
        // Exactly U descriptor initializations/pushes, one finish, no default
        // padding slots/clones. Incomplete finish consumes/drops prefix. Any
        // failure drops names and descriptor prefix/backing before caller pools,
        // then preparation/publication charge. Nested element allocation is a
        // caller obligation; these Record fields own no nested heap backing.
        let entries = entries.finish().map_err(exact_error)?;
        Ok(Records { entries, names: PackedString(names) })
    }
}

fn prepare(geometry: GeometryOwner) -> Result<MetadataPreparation, Error> {
    // First operation: no header/workload/plan read, issuance, debit or allocation.
    // ENTRY: quote_metadata_entry calls extended_metadata_quote once, with its
    // Result::map/map_err, then this map_err/propagation to Binding. Current
    // refusal drops incoming geometry (original, experiment, limits, controller,
    // root in declaration order); no MetadataPreparation is assembled.
    // Future success would call original() once, enter_metadata once, map its
    // Result, move permit+geometry into preparation, then return/destination.
    // enter_metadata currently only matches an uninhabited quote: there is NO
    // permit construction/Arc clone/reservation/debit implementation to count.
    // Parent must fund COMPUTING extended quote + bootstrap + entry + permit
    // construction, original Context Arc retain/release, controller and eventual
    // retirement BEFORE issuance. No existing geometry charge supplies that
    // missing payment. Future quote/issuance failure drops incoming geometry;
    // future success must account for all argument/Result/assembly/move overlaps.
    let quote = original_source::quote_metadata_entry().map_err(|_| Error::Binding)?;
    let permit = geometry.original().enter_metadata(quote).map_err(|_| Error::Binding)?;
    Ok(MetadataPreparation { permit, geometry })
}

fn limits(geometry: &GeometryOwner) -> Result<checker::Limits, Error> {
    // After genuine funded entry AND body payment only: original/header borrows,
    // cached Header scalar reads, checked text+key sum, json_values*32 and sum,
    // depth+1, max_work conversion, Limits assembly/Result propagation. No plan
    // traversal is used to quote outside parent funding. Arithmetic/conversion
    // failures return Bounds and retire preparation through collect's caller.
    let header = geometry.original().header();
    let strings = plus(header.text_bytes(), header.key_bytes())?;
    Ok(checker::Limits {
        max_operations: header.operations(), max_inputs: header.input_refs(),
        max_dimensions: header.dimensions(), max_rank: header.dimensions(),
        max_string_bytes: strings, max_attribute_nodes: header.json_values(),
        max_attribute_bytes: plus(strings, header.json_values().checked_mul(32).ok_or(Error::Bounds)?)?,
        max_attribute_depth: plus(owned_backing::MAX_JSON_DEPTH, 1)?,
        // Existing limits, not a fresh per-selection work budget.
        max_work: usize::try_from(geometry.limits.max_work).map_err(|_| Error::Bounds)?,
        max_scratch_bytes: MAX_PROGRAM_BYTES,
    })
}

fn collect(mut preparation: MetadataPreparation) -> Result<MetadataOwner, Error> {
    // SELECT once: original() borrow; select performs one original.root() lookup,
    // one Arc::ptr_eq (no clone), selected-bit read, then check_usable's ledger
    // Option/lock/burn check and lock retirement, then one selected=true write.
    // Mismatch skips later checks; already-selected skips usability; missing,
    // poisoned or burned ledger skips mutation. Each maps to Binding here and
    // drops preparation: permit/PrimitiveAttempt before geometry. Attempt drop's
    // uncommitted sticky-burn lock/poison handling, controller and Context Arc
    // release need lifecycle payment; there is no refund/reset/commit here.
    preparation.permit.select(preparation.geometry.original()).map_err(|_| Error::Binding)?;
    // BODY FUNDING BARRIER. No value of this type exists, even under cfg(test).
    // Prospective inventory must cover checker.work_ceiling PLUS initial quote
    // AND check's SECOND quote, init(N,R,F,B), copy(U,T), full B UTF-8 validation,
    // all retirement/errors and wrapper/assembly moves BEFORE next statement.
    let _body = pllm_models::source_work::extended_metadata_quote().map_err(|_| Error::Binding)?;
    let limits = limits(&preparation.geometry)?;
    // QUOTES: one initial checkpoint_requirements::quote; check below calls
    // quote once more. Each includes Traversal::resources_for_native_slots(F).
    // In check's census, per visited operation (<=N), Traversal::new_native
    // repeats resources_for_native_slots + check_resources before its walk.
    // Early failure runs only a prefix. Existing checker work_ceiling covers
    // checker checking, NOT caller initial quote, entry, pool init, copy or
    // retirement. Both top-level quote paths and repeated per-op resource checks
    // must appear in independent inventory, without double-pricing checker work.
    let quote = checker::quote(limits).map_err(|_| Error::Binding)?;
    if quote.work_ceiling > limits.max_work || quote.scratch_bytes > limits.max_scratch_bytes {
        return Err(Error::Bounds);
    }
    let text_bound = preparation.geometry.original().header().text_bytes();
    // Prequote Core layouts BEFORE any allocation. Layout::array validates
    // extent/alignment/isize bounds, unlike byte multiplication alone. Three
    // typed pool layouts + R descriptor layout + B byte layout; Core new later
    // repeats its own check. Helpers confer no source/funding authority.
    let operation_bytes = exact_buffer::layout_bytes::<checker::OperationSlot<'_>>(
        quote.operation_slots,
    ).map_err(exact_error)?;
    let requirement_bytes = exact_buffer::layout_bytes::<checker::Requirement<'_>>(
        quote.requirement_slots,
    ).map_err(exact_error)?;
    let frame_bytes = exact_buffer::layout_bytes::<NativeFrame<'_>>(
        quote.attribute_frame_slots,
    ).map_err(exact_error)?;
    let pool_bytes = plus(plus(operation_bytes, requirement_bytes)?, frame_bytes)?;
    let descriptors = exact_buffer::layout_bytes::<Record>(quote.requirement_slots)
        .map_err(exact_error)?;
    let name_bytes = exact_buffer::layout_bytes::<u8>(text_bound).map_err(exact_error)?;
    let walker_controls = quote.attribute_workspace_bytes.checked_sub(frame_bytes).ok_or(Error::Bounds)?;
    let backing = plus(descriptors, name_bytes)?;
    // A complete typed-control/move/drop overlap inventory has NOT been priced.
    // Refuse rather than pretending this subtotal is a 64 MiB phase certificate.
    let controls = complete_controls()?;
    let workspace_payload = plus(controls, plus(walker_controls, pool_bytes)?)?;
    // KNOWN phase: current root usage + workspace payload/guard + publication
    // backing/full wrapper. size_of<MetadataOwner> includes Publication inline,
    // padding and its guard ONCE. try_build adds wrapper-minus-guard inline to
    // publication charge; workspace guard remains separate. No fabricated root
    // subtraction or repeated geometry heap census. Checked sums, root usage,
    // 64 MiB comparison and admit_peak all precede external charges. Unknown
    // typed controls/coexistence still block full peak certification.
    let publication_bytes = plus(backing, size_of::<MetadataOwner>())?;
    let workspace_bytes = plus(workspace_payload, size_of::<OwnedExternalCharge>())?;
    let phase = plus(preparation.geometry.root.budget().used_bytes(),
        plus(workspace_bytes, publication_bytes)?)?;
    if phase > MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
    preparation.geometry.root.budget().admit_peak(phase).map_err(|_| Error::Bounds)?;
    // CHARGES (prospective only): two OwnedBudget::charge_external calls. EACH
    // checked-adds its OwnedExternalCharge guard, reserve_bytes, clones Budget
    // Arc once, assembles/returns guard; overflow/reservation failure creates no
    // guard. First failure drops preparation; second drops workspace then
    // preparation. Both guards' release_bytes/Arc drops need event inventory.
    // Core adds ZERO external charges/Arcs. These two are the only new external
    // charges: workspace transient, publication retained. No refund/new ledger.
    let workspace = preparation.geometry.root.charge_external(workspace_payload)
        .map_err(|_| Error::Bounds)?;
    let publication_charge = preparation.geometry.root.charge_external(backing)
        .map_err(|_| Error::Bounds)?;
    let mut primary = None;
    let failure = &mut primary;
    // Full Publication inline/padding admitted by try_build BEFORE any pool.
    // Move preparation into builder; borrow geometry only INSIDE, ending every
    // view/pool borrow before assembling Publication. No outside borrow crosses
    // this move. Captured preparation and builder locals retire before charge
    // on builder error/unwind; workspace remains until try_build returns.
    // try_build: checked wrapper-minus-guard and payload sums; resize_payload's
    // checked guard sum/positive reserve_bytes (or unchanged-size branch); invoke
    // builder; assemble OwnedCharged/Result. Overflow/resize failure skips builder
    // and drops captured preparation plus publication charge. Native-only error
    // maps to Bounds; logical builder error is captured/restored through primary.
    let result = publication_charge.try_build(move |budget| {
        let built = (|| {
            let records = {
                let geometry = &preparation.geometry;
                // Five Core allocations total on nonempty success: N/R/F pools,
                // U descriptors, B bytes. Each new checks layout, zero-size path
                // skips allocator, otherwise allocation-null maps to Bounds;
                // allocator abort is not recoverable. No resize/reserve/copy on
                // finish. Core owns each initialized prefix on failure/unwind.
                // Pool loops call empty() and push ONCE PER SLOT: N, R, F calls,
                // no seed cloning/Vec resize. Push refusal returns incoming T;
                // explicitly drop it before Bounds, then prefix owner retires.
                let mut operations = ExactBuffer::new(quote.operation_slots).map_err(exact_error)?;
                for _ in 0..quote.operation_slots {
                    if let Err(value) = operations.push(checker::OperationSlot::empty()) {
                        drop(value);
                        return Err(Error::Bounds);
                    }
                }
                let mut operations = operations.finish().map_err(exact_error)?;
                let mut requirements = ExactBuffer::new(quote.requirement_slots).map_err(exact_error)?;
                for _ in 0..quote.requirement_slots {
                    if let Err(value) = requirements.push(checker::Requirement::empty()) {
                        drop(value);
                        return Err(Error::Bounds);
                    }
                }
                let mut requirements = requirements.finish().map_err(exact_error)?;
                let mut attributes = ExactBuffer::new(quote.attribute_frame_slots).map_err(exact_error)?;
                for _ in 0..quote.attribute_frame_slots {
                    if let Err(value) = attributes.push(NativeFrame::empty()) {
                        drop(value);
                        return Err(Error::Bounds);
                    }
                }
                let mut attributes = attributes.finish().map_err(exact_error)?;
                // Each pool has one finish; Incomplete consumes/drops prefix.
                // Later-pool failure drops earlier finished Boxes; no callback
                // here allocates nested storage. Any changed initializer would
                // need its transitive allocation/init/retirement inventory.
                // check performs quote AGAIN; per-op native resources repeated
                // as above. Preserve numeric/tensor/dedup/conflict contracts.
                let view = checker::check(geometry.plan(), limits, checker::Scratch {
                    operations: &mut operations, requirements: &mut requirements,
                    attributes: &mut attributes,
                }).map_err(|_| Error::Binding)?;
                // Checker refusal erases diagnostic to Binding, drops pools,
                // then preparation through builder. No copy/view on failure.
                // Distinct attribute values emitted once; dedup only removes;
                // tied heads and phases are separate header occurrences. T<=B.
                Records::copy(&view, text_bound, quote.requirement_slots)?
                // View and all borrowed pools retire before moving preparation.
            };
            budget.admit_peak(budget.used_bytes()).map_err(|_| Error::Bounds)?;
            Ok(Publication { records, preparation })
        })();
        built.map_err(|error| {
            // Partial Records/pools/preparation have retired; preserve actual
            // compiler primary, not a native Resource substitute. A panic skips
            // translation and unwinds owned scopes before publication charge.
            *failure = Some(error);
            pllm_garble::shared_allocation::Error::Resource
        })
    });
    // Workspace spans try_build, all failures and publication assembly, not its
    // later retained lifetime. Records -> preparation (permit -> geometry) ->
    // enclosing charge on later ordinary drop/unwind. Attempt Drop sticky-burns,
    // then its root Arc/controller fields retire; geometry still retains source
    // custody. Publication's retained inline is not borrowed from workspace.
    drop(workspace);
    result.map_err(|_| primary.unwrap_or(Error::Bounds))
}

pub(super) fn complete_controls() -> Result<usize, Error> {
    // CONTROL ASSIGNMENT (code-path owners, NOT proof of funded execution):
    // Item                       | Current assignment / missing assignment
    // Existing Geometry/root     | Existing root/controller/original charges;
    //                            | root.used_bytes already includes these. Do
    //                            | not charge their backing again for aliases.
    // Transient pools/walker     | Workspace payload includes exact N/R/F typed
    //                            | backing; walker_controls excludes F backing;
    //                            | controls UNKNOWN. Lives through try_build.
    // Workspace guard            | Own explicit size_of<OwnedExternalCharge>,
    //                            | charge_external reserve + one Budget Arc.
    // Publication backing       | R descriptor maximum + exact B name bytes,
    //                            | even though actual descriptors use U<=R.
    // Publication inline/guard  | Full size_of<MetadataOwner> includes guard;
    //                            | charge_external adds guard + one Budget Arc,
    //                            | try_build adds wrapper-minus-guard inline.
    //                            | Publication retains Records + preparation;
    //                            | payload retires BEFORE this guard. No third
    //                            | charge/Arc from any Core buffer.
    // Permit entry/lifecycle     | Retained inline now has publication owner,
    //                            | BUT no metadata issuance/payment exists.
    //                            | Parent extended-quote/bootstrap/permit work
    //                            | and controller funding still UNIMPLEMENTED.
    // Unknown transient controls | limits/quotes/Scratch/view; Core prefix/Box,
    //                            | Vec/String/from_utf8-error descriptors;
    //                            | iterators/borrows/ranges/arithmetic/results;
    //                            | primary/closure, guards/assembly/move/drop
    //                            | overlaps still require explicit inventory.
    // Return/overlap controls    | Publication, OwnedCharged, preparation,
    //                            | Result/return/destination simultaneous slots
    //                            | beyond retained inline remain UNKNOWN; not
    //                            | paid by released workspace after return.
    // All new charges above are unreachable planned paths, not current funded
    // admission. Source-event counts still need literal helper/branch prices;
    // no E/W constants or whole-lifetime work payment follows from byte helpers.
    // Full peak must include current root, BOTH guards/controllers, typed
    // Results/moves, walker/pools, descriptors/packed String and partial owners
    // with phase overlaps, without duplicate existing charges. NOT certified
    // <=64 MiB. Core's source contract fixes logical array lengths (no
    // try_reserve_exact overcapacity branch); Box->Vec->String preserves B
    // backing, and checked appends cannot grow. This is NOT allocator-rounding,
    // RSS, transitive-element-allocation, CPU or lifetime qualification. Full B
    // UTF-8 validation, prefix/drop/finish, error-owned Vecs and all helper/branch
    // costs remain UNPRICED. No E/W constants, issuer or funding added.
    Err(Error::Bounds)
}
