# D36 private one-row records (review before execution)

These records bind the existing selected arithmetic, not authenticated source,
transport, checkpoint conformance, generic fanout or joint privacy. No runtime
hash is reported as executed by this source definition.

## Framing

Reuse `typed::digest`: each part is `u64LE(byte length) || bytes`, in order.
The domain part is the exact byte literal `pllm.sigma.causal-softmax-row.v1`;
neither this domain nor subsequent `.v1` tags have a final NUL.
Integers below are fixed little endian; counts/positions are public. Variable
arrays contain only their declared live prefix, not padded slots.

`declaration.v1` parts:

1. ASCII `pllm.sigma.causal-softmax-row.v1`, `declaration.v1`.
2. Exact equation string from `causal_softmax_spec::EQUATION`.
3. Original SourceIdentity checkpoint/profile/implementation (three 32-byte parts).
4. Original plan, session, operation and next (four 32-byte parts).
5. Source paper SHA ASCII
   `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`
   and source commit ASCII `aa0f2f860266fb79901405ef5e9965b8c9fe2e88`.
6. Original input and output edge digests, public row width as u64LE,
   active count u8, then exact ordered active-position bytes.
7. Signed38 lower/upper `-2^37,2^37` as i128LE; fixed geometry bytes
   `[50,12,39,14,12,6,8,6,0,50,24,12]`.
8. Actual certified E0/E1 raw u64LE payload digests, actual verified inverse raw
   payload digest, and inverse source-lookup numeric-contract digest.
9. Exact live ordered outer ConsumerUse digests. For k=2/3: 13k+1. Child keeps
   its separate original eighteen-use declaration and eleven-stage contract.

`inverse.numeric.v1` uses the same domain/equation/source pins, E0/E1/inverse
payload pins and lookup geometry `[8,0,6,50,12]`, plus literal sentinel equation `T[0]=0` and literal
nonzero equation `T[z]=floor(2^18/z)`. No role, mask or operation enters the
payload hash; actual source_lookup table identity still incorporates numeric
contract and its original invocation binding incorporates real context.

`wire.v1` parts: domain, tag, original input wire, operation, u16LE slot.
Derived wire names are metadata only. The actual original root/mask/native
child owners and private lane mappings provide authority, never that hash.

Uses are the existing core `pllm.consumer-use\0v1\0` records. There is NO new
outer use/ledger/region hash or packet. Actual original runtime bindings are
appended by the private dealer at their protected reads (consumed prefix cannot
change), then frozen into original per-role ledgers before handles publish.
Max updates are eight reads each; differences two per active lane;
child transfer/receipt; sum k reads then projection/TR/coordinate/lookup; k E/R
pairs; product-to-GapARS and final output. Burn before protected reads/checks.

`stage.v1` parts are domain, tag, declaration digest, u16LE factory ordinal.
Max comparison/Select pairs use ordinals1..2(k-1); inverse/product/Gap use
2(k-1)+2/+3/+4. Coordinate keeps A's ORIGINAL binding/Q6-to-Q0 bridge context,
with the parent's logical operation consumer. The child keeps its ORIGINAL
separately bound D29 contexts/records. Outer transfer is a parent operation use;
the transferred original Binding is precommitted to the acyclic child operation,
and the child enforces that consumer in its own original eighteen-use scope.
Each native Context also binds actual original
operand/output wire identities; no runtime Binding is replaced by metadata.
Native peer codecs are unchanged; no new packet format.

## Range and transcript facts

Whole original score slice passes signed38 admission before proof/hash/operation
entropy. Max difference stays signed39. Preserve full ring50 d for Select50;
never sign-extend a masked ring39 view. Max-relative differences are in [0,2^38).
Inherited D29 receives original difference owners, never clear recovered data.

Some E is exactly4096 by the max relation. Thus 4096<=D<=4096k, unsigned14
projection is exact, Z=TR14,6(D) is raw unsigned8 in64..192. Q0 view retains
actual Q6 native owner/issuance. Inverse table's global maximum262144 does NOT
mint R<=4096: only this genuine coordinate relation gives1365<=R<=4096.
Product<=2^24, well inside sole final GapARS50 quarter ring. No renormalization.

k=1 uses charged public4096/inactive-zero custody, zero private exchanges.
k=2/3 uses 20/22 exchanges and40/44 original native peer frames. Public inactive
positions never enter secret max/E/sum. Outputs remain closed until both Ready.
Source/counter/frames remain resident to actual drop; cancellation is not refund.

Native execution, literal canonical vectors, full owner/peak/work accounting and
independent lifecycle review remain acceptance gates. This text is not a golden
hash, execution result, privacy proof, SDK registration or model-quality claim.

## Admitted logical phases

The real original working charge covers bootstrap/query controls before native
splits. The constructor rechecks immutable source capacities, pointer-equal
Budget identity, the exact total and cumulative work before proof/factories.
Neither returned numeric records nor source/table labels grant authority.

No peak-retirement credits: root masks/two values; maximum DReLU39/Select50
parties/masks; actual inherited child; coordinate/inverse/product/Gap owners;
source/keepalive Arcs; two state/slot Arcs and shared coordination; all native
frames plus parent wrapper maxima; actual parent difference/broadcast/one-word
sum buffers; and concrete call/result/old-new transitions are summed. Empty
`Box<[u64]>` replacement retires a zeroized buffer once without allocation.
Original input Vec capacity and its guard stay additional to working peak.

Work categories are logical source-level events, not CPU/time: whole validation,
actual length+capacity input retirement, complete fixed record initializations
(including padding), ordered uses, lane/initialization/erasure visits, original
native AES/random-byte bounds and fixed metadata-query events. Parent AND child
512-floor passes are disjoint; both child evaluator costs count. Public table
bytes include prep, pin and native full-table rehash, not merely final digests.
No unused child slice is lent to another phase. A short total is rejected before
proof/entropy/factories; zero work rejects before the constructor quote pass.

Known logical hash controls include nested core edge/use workspace and active
record/hash slots. Dependency-private SHA/AES/helper stacks, instruction counts,
allocator overhead and RSS remain unqualified, never zero. Existing primitive
owner resource/work contracts are inherited, not independently recertified.
Raw peer mailboxes and callback-owned work/storage are caller admission gates.

After both child own-Ready receipts, E remains in original child slots. Normal
receipt cleanup does NOT abort the child; the parent retains its real coordinator
and mask. Abnormal parent Retire calls the closed downstream abort hook after
loans release, even when both child Roles were consumed. No parent backreference
or external callback exists under these nested locks. Public parent output still
requires both parent Ready; root/coordinator/output drop revokes siblings.
