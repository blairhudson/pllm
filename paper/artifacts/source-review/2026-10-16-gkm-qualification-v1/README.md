# Quarantined GKM source qualification — 16 October 2026, v1

**Documentary evidence only.** Upstream code is specification/oracle text, never
built, executed, imported or linked. This archive contains no model/checkpoint,
upstream build tree, executable upstream implementation or benchmark result.
The [qualification card](../../../COMPETITOR-QUALIFICATION.md) records conclusions
and unresolved gates. Source and code access do not qualify a complete comparator.

## Records and conventions

- `repository-record.json`: dated, bounded fields from GitHub repository/owner
  metadata, exact retrieval URLs/headers and whole-response hashes. These mutable
  fields identify the repository and ePrint relationship; they are not a frozen
  evaluated-build claim or a complete license/dependency audit.
- `code-lock.json`: commit `271637be7196d26a5433ca1e62b1a5357e769a5b`, tree identity,
  twelve selected file SHA-256/Git-blob hashes, byte/line counts, per-file GET
  provenance, exact source locators, and bounded line extracts with their own
  hashes. Full upstream bytes were hashed in memory; only excerpts were retained.
  Upstream `LICENSE` is preserved in the extract record. Original authors retain
  attribution; no upstream code enters PLLM's runtime or conformance implementation.
- `primary-locators.json`, `gkm-pdf-page-*.txt`: original frozen PDF identity,
  existing `pdftotext` binary/version/hash, exact stdout extraction commands and
  bounded relevant PDF pages 6–13. Follow existing source-review conventions:
  select pages with `-f/-l`, preserve original extracted text, distinguish the PDF
  hash from text hashes. Fig.3's apparent typo is not silently corrected in text.
- `local-locators.json`: hashes and citation ranges for current PLLM source bytes,
  not installed-build identity. A source change requires a new qualification pass.
- `receipt.json`: documentary artifact and reviewed-document hashes. It is an
  integrity inventory, not a signature, execution attestation or security proof.
  The prior manifest is extended by a dated record; its original fields are
  separately canonical-hash checked. Older source reviews remain historical
  snapshots and are not rewritten to conceal their original manifest hashes.
- `capture.py`: create-only documentary acquisition driver. It uses HTTP GET,
  local byte reads and the existing PDF text tool. It never runs source text.
  Initial capture bounds for LICENSE/async-verifier extracts were corrected before
  sealing; the successful capture and separate metadata acquisition are recorded
  in the receipt. Reuse in a new dated directory, not over this frozen archive.
- `check.py`: independent read-only byte/hash/JSON/locator checker. It imports only
  Python's standard library, never `pllm` or upstream modules. It writes no files.

## Reproducible read-only checks

From repository root, Python 3.11+:

```sh
python3 -B paper/artifacts/source-review/2026-10-16-gkm-qualification-v1/check.py
python3 -B paper/artifacts/source-review/2026-10-16-gkm-qualification-v1/check.py --online --pdf-text
```

Default: strict JSON (including duplicate/non-finite rejection), archive and
document hashes, unchanged frozen primary records, original PDF hash, stored
page/extract hashes, local source hashes, code-citation ranges and local links.
It does **not** claim to reconstruct full upstream files from partial excerpts.

`--online`: re-fetch immutable-commit raw source files into memory; compare full
SHA-256, Git blob identity, lengths and exact excerpt slices; confirm commit/tree.
Mutable metadata/API headers need not match historic response hashes. Retrieval
failure is an unresolved recheck, not evidence that code never existed.

`--pdf-text`: rerun the existing pinned PDF text utility to stdout and compare
page bytes; no PDF/build/model/provider/operator execution. Another tool binary
requires a separately recorded extraction review rather than replacing old text.

The checker validates citation structure and source identity; manual source
reading supports the interpretation. Neither command proves protocol security,
whole-artifact correctness, exact Qwen parity, memory eligibility or performance.
All future conformance/model/measurement execution stays parent-owned and serial.
