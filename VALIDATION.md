# Validation for 0.17.0a1

## Executed on this source tree

Local validation on macOS ARM64 used Python 3.11.15, 3.12.12 and 3.13.15 with
the repository-pinned Rust 1.85.1 toolchain.

| Check | Result |
| --- | --- |
| Python suite with the compiled Rust backend | 249 passed on each supported Python version; 16 HE/SDK tests deselected |
| Python suite with the explicit numeric reference | 247 passed; 18 Rust/HE/SDK tests deselected |
| Real TenSEAL tests | 13 passed |
| Installed OpenAI SDK tests | 4 passed |
| Independent lifecycle study tests | 45 passed |
| Cargo workspace tests | 10 passed across 4 suites |
| Cargo correctness lints | Passed; 2 non-correctness style warnings |
| Maturin development build | ARM64 abi3 wheel compiled and installed |
| UV distribution build | Source archive and ARM64 abi3 wheel passed metadata/content checks |
| Documentation tests | 13 passed |
| Documentation dependency audit | 0 vulnerabilities |
| Fumadocs content and internal links | 29 pages, no errors |
| Fumadocs typecheck and static build | Passed; 33 routes generated |
| Python source and TOML parsing | Passed |
| Workflow structure and pinned action checks | Passed |
| Source manifest and SHA-256 integrity | 457 files passed |
| TypeScript syntax transpilation | 11 files passed |
| Public exports and module entry point | Passed |
| Python source relocation outside the checkout | Passed |
| Clean installed-wheel smoke tests | Passed on Python 3.11 and 3.13 outside the checkout |
| Paper build | Pandoc Markdown produces the seven-page, two-column PDF |

## Not executed here

The cross-platform wheel matrix, native benchmark comparisons, remote publication,
physical wide area networking and a full model checkpoint were not executed. No
new speedup measurement is claimed by these build checks.

The configured native CI jobs require the compiled Rust backend. They test
installed platform wheels outside the checkout. The HE job runs isolated cases
and rejects skipped tests. PyPI publication requires committed dependency locks
and successful verification jobs. Documentation deployment requires the framework
build to succeed.

## Records

The `verification` directory retains the initial migration logs. Prior release
logs are not presented as current validation. `MIGRATION.md` and `PROVENANCE.md`
identify the source and changes.
