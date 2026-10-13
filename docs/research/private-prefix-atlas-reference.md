# Privately retrieved public-prefix atlas: conditional screen, not admission

H3 in `docs/plans/low-tier-all-link-hypotheses.md` (ranked candidate #4).
No model, native provider, timing/RSS benchmark, build or download performed.
Evidence: `docs/evidence/research-private-prefix-atlas.json`. Complete-cost verdict:
**unknown; retain only for high-reuse public catalogs with short client horizons**.
Existing local capsule hits win; ordinary fresh private prefixes receive no benefit.

## Concrete construction and knowledge

Trusted publisher independently approves **explicitly public** token sequences,
computes completed canonical prefills, and publishes existing full-KV capsule
bytes. Never solicit, discover, log or publish private prompts, private branches,
private suffixes or their states. Public publisher knows entire catalog, source
and numeric contracts, not client choice. No finite-task response lookup: capsule
contains exact prefix KV and last logits; arbitrary suffix/generation still execute.
Private embedding HE and existing response/DPF toys are separate controls.

Each catalog epoch has N templates of one common public prefix length and admitted
geometry, one compiled workload/source/numeric identity, maximum capsule size S.
Pad each capsule to P=ceil(S/65536)*65536, including final zero padding. Stripe j
is a separate N-by-65536-byte table. Publisher transmits **every stripe to both**
Preparation and Inference, which own independent replicas. Each stripe satisfies
native N<=262144 and N*65536<=512 MiB; this layout therefore caps N at 8192.
Aggregate server storage is 2NP, not one free shared replica. No page/delta sharing
used: context-graph private AEAD pages are caller-owned and cannot become public
catalog objects. Public exact delta graphs would require separate complete pricing.

Client gets authenticated entire public catalog metadata, resolves chosen public
template locally, and generates fresh independent native `private_page_issue`
keys for the same private index in **each** of m stripes. Never disclose index
through digest/length fetches, capsule path names, selective metadata or URLs.
Client sends one share to each role; both evaluate their entire stripe in fixed
public access order, reply with 65536 bytes plus 65-byte header. Client XORs replies,
writes stripes sequentially into caller-private staging file, removes authenticated
padding after verifying declared length, then verifies original capsule SHA-256
against publisher commitment before any import. No direct selected-server fetch.

Native DPF is binary tree with fresh 128-bit roots/id, AES-128 stretch and leaf
bits, 17-byte correction word per depth. Query size is 82+17*ceil(log2 N), reply
65+65536. Two queries and two replies per stripe. DPF/FSS material is included
inside query bodies, not another additive charge. Descriptor binds table digest,
record count and width. Keys consumed once, decoder burns even on failure; fresh
retries incur full new charges. Dense uniform XOR alternative is explicit:
q0 uniform N-bit vector, q1=q0 XOR e_i; answer_b=XOR_j(qb[j]*record_j).
This gives perfect single-server index privacy but sends 2*ceil(N/8) query bytes
per stripe plus authenticated bindings (65-byte headers plus F per frame in this
budget). Independent uniformly random bit vector for every stripe/request; unused
tail bits fixed zero; no PRG/security claim needed for algebraic construction.
Same replica deliveries, fixed scans, replies, metadata and import costs; evidence
prices dense alternative separately. Script exhausts this tiny alternative,
**not** native AES/DPF security, randomness, constant-time behavior or composition.

Topology must use genuinely separate non-colluding operators, no common proxy,
logging plane, administrator, issuer or key owner receiving both shares. Publisher
does not issue private queries. Client owns random roots, private index, replies,
plaintext reconstructed capsule, trusted publisher root, local cache and private
suffix KV. Servers know public catalog, public epoch/shape, timing and their share;
collusion reveals index. Publisher commitments authenticate approved computation,
not servers' self-asserted hashes. Malicious server corruption detected by whole
capsule commitment; availability, selective failure and side-channel privacy are
not proved. Fixed retries/abort handling required; never expose digest failures
with selected index. Authenticated encrypted links and independent cryptographic
review remain implementation gates, not existing API features.

Run retrieval **before** online decoder execution. Preparation may scan immutable
public atlas during this setup phase, then erase private query artifacts; it must
not retain/correlate them with later offline private correction issuance. During
online execution Preparation remains idle under existing protocol. Inference sees
only its share and ordinary masked decoder traffic. Non-collusion must cover atlas
and later protocol logs for entire retention lifetime, not merely request overlap.
An epoch's per-stripe ledger allows 4096 queries, process-local only. Distributed
durable burn/restart and retirement need new contract; restart cannot silently
reset replay ownership. Batch cap 64 does not cap m: sequential queries work.

## Exact identity, lifecycle and scope

Whole capsule digest/size come independently from publisher. Admission calls
existing import with original bytes: source/numeric `_state_numeric_contract`,
`prefix_f32`, unverified ordinary prepared full-KV ownership, compiled continuation
contract, public IDs, complete finite little-endian f32 KV/logits, dimensions and
capacity. Native binding/provenance and cache key additionally bind compiled
digest, bundle fingerprint, IDs and causal reduction. Manifest must bind source
lock, body fingerprint, numeric digest, compiled/workload identity and epoch;
reject wrong epoch/source/geometry/digest before installing cache. Trusted catalog
root is client-held; server cannot roll client back. Source change invalidates all.
Only PUBLIC prefixes permitted; subsequent private state remains local and is
never an atlas update. Exact greedy response equality alone insufficient:
full logits and all active KV must match ordinary execution for suffix/decode.

Cold epoch: charge publisher source acquisition, N prefills, serialization/hash,
both catalog transmissions/admissions and manifest distribution. Cold client:
authenticate metadata, retrieve once, stage/authenticate/import bounded capsule,
then use fresh one-use prepared suffix/decode material. Warm same-capsule client:
existing local cache does everything already; PIR should skip and has no benefit.
Warm miss: new m queries/replies/import and eviction work. Full-catalog disk cache
may stream one capsule at a time with small RAM; do not equate NP disk bytes with
whole-client resident memory. All disk owners/file cache charged separately.

**Privacy scope:** fixed retrieval shapes hide selected index only. Common prefix
length/continuation shape mandatory. Current cache hits and reuse/demand row counts
remain visible; skipping PIR leaks hit/miss. Either explicitly allow these leakage
patterns, or issue identical dummy retrievals on every scheduled request and pad
decoder shapes/rows consistently. Dummy retrieval still costs full replies/scans.
No end-to-end private-template claim follows without that composition proof.
No extra local body MACs for retrieval, but whole execution must retain >=90%
remote body MACs independently prefill/decode; existing controls still required.

## Complete pricing and break-even

Probe's **stipulated** metadata budget M=256+128N+48m: global signed identity/root,
N fixed digest/length/public-ID commitment descriptors, m table digest/dimension descriptors.
Trusted publisher signature included in 256-byte header; token lists already in
capsules. Local index resolution uses approved public template mapping; if mapping
does not fit stipulated descriptors, add its complete distribution/storage bytes.
Per-message authenticated framing budget F=64 is sensitivity assumption, not a
serialized transport measurement. Signing keys preprovisioned independently;
non-preprovisioned credentials and all handshakes add unknown setup bytes.

Let K clients, H requests/client, T aggregate real misses or cover retrievals:

* A=2(NP+M+F)+K(M+F): both replica deliveries plus each client's metadata.
* R=m[2(82+17ceil(log2 N))+2(65+65536)+4F]: each retrieval.
* Atlas known incremental traffic = A+TR, plus unknown source/setup/retries/wire.
* Strong local full-catalog control = K(NP+M+F), once per client/epoch, then zero
  catalog network on every hit. Disk cache may be entirely viable at Low.
* Download-all per request = KH(NP+M+F); weaker than retained catalog.
* Prepositioned one-capsule cold = K(P+M+2F); local capsule hit zero network.
  Manifest could be narrower for this control, making it stronger still.
* Ordinary private prefill = common lifecycle + sum B_i per request. Atlas uses
  common suffix/decode lifecycle plus A+TR and publisher upstream work. B_i must
  be measured on strongest completed **Low-qualified** control, not guessed.

Catalog comparison requires T < [KNP-2(NP+M+F)]/R. Negative RHS rejects.
One cold miss/client requires K > 2(NP+M+F)/(NP-R), when NP>R.
All-miss or mandatory-cover horizon requires
H < [KNP-2(NP+M+F)]/(KR). Strict inequalities; exact integer threshold in probe.
Against ordinary prefill require sum avoided B_i > A+TR plus differential unknown
traffic. Against local capsule hit there is no positive byte break-even.

IID probabilities p_i and static top-C cache: expected misses/client are
sum_topC[1-(1-p_i)^H]+H*(1-sum_topC p_i). Full-catalog C=N uses expected unique
templates only; one-slot LRU needs actual trace/transitions, not this static model.
Probe includes uniform and 90%-hot sensitivities. Access distribution is externally
specified workload assumption, never discover private template traffic to fit it.

S=2,967,627 comes from pinned publisher record; m=46, P=3,014,656. N=32, K=16,
H=1: 290,056,128 known atlas bytes vs 1,543,609,856 retained-catalog bytes
(81.21% **catalog-only**, not whole-response reduction). Cold threshold K=3;
all-miss horizon H<=13 wins known bytes. H=16 all misses loses against retained full catalog
despite beating download-all. N<=2 cannot repay two replies against full catalog
in this model. Historical 215,002,387 covered-byte prefill delta is **not** matched
Low/all-link B_i, and combined publisher CPU 8.431498 s is not N times per-template
compilation cost. Neither substituted into complete-cost verdict.

Publisher N prefill CPU/serialization, both server admission/hashing and DPF tree
expansion, 2TNP logical scan bytes, client RNG/query issuance, 2TP replies/XOR,
staging disk writes/reads, SHA-256/signature validation, parser/copies/cache import
all stay charged. CPU quantities and physical DRAM/disk/wire are null until measured.
No free publisher or replicas; scan bytes are logical work, not network traffic.

Existing importer reads whole artifact, msgpack materializes binary values, then
copies KV/logits and cache snapshots. Sequential PIR only bounds page scratch,
not this import peak. P+M below 256 MiB is **not** whole-client qualification.
Payloads, cache copies, runtime, allocator, mapped files, file cache/disk owners,
authentication buffers and numeric workspace must fit conservative admission plus
fresh-client worst peak <=268,435,456. Whole-client peak, seven serving metrics,
complete-cost gain and remote-body ratios remain unknown.

## Replay and remaining gates

```sh
target/frontier-sdk-env/bin/python scripts/probe_private_prefix_atlas.py --output docs/evidence/research-private-prefix-atlas.json
target/frontier-sdk-env/bin/python -m pytest -q tests/test_private_prefix_atlas_probe.py -o pythonpath=.
target/frontier-sdk-env/bin/ruff check scripts/probe_private_prefix_atlas.py tests/test_private_prefix_atlas_probe.py
```

Evidence hashes implementation, tests, note, native bindings/primitive, capsule
implementation and source-locked references. Deterministic JSON; no timestamps,
random private indices, prompts, keys or private states. Tiny XOR oracle verifies
every target/random share and complete one-server query/reply histogram; collusion
oracle confirms expected failure. Tests also cover padding, limits, complete links,
strict thresholds, strong long-horizon control and IID oracle.

No valuable native full-width run authorized here. **Parent-only** next step,
after implementation/admission/privacy review: add integrated atlas candidate to
ordinary benchmark under existing `fresh_workers` exclusive lock/admission/swap
guard, offline pinned source, alternating strongest Low controls, >=3 fresh repeats;
run full logits/all-KV generated Qwen2/Qwen3/Llama gate first. Suggested command
shape (new integration/worker recipe required, deliberately not executable today):
`fresh_workers: python scripts/probe_private_prefix_atlas_sdk.py --processes 3
--catalog 32 --clients 16 --horizon 1 --offline --output <fresh-evidence.json>`.
Do not launch standalone operators as substitute for whole-client/SDK proof.
