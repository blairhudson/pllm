"""Ordinary consumer-lock identity gate; no Cargo, downloads, or native work."""

import copy
import json
from pathlib import Path
import tomllib
import unittest


HERE = Path(__file__).resolve().parents[1]
FROZEN = json.loads((HERE / "bounded-allocator-contract.json").read_text())
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
GIT = (
    "git+https://github.com/poulpy-fhe/poulpy?rev="
    + FROZEN["poulpy_commit"]
    + "#"
    + FROZEN["poulpy_commit"]
)
EXPECTED = [
    dict(package, source=REGISTRY)
    for package in FROZEN["solver_dependency_plan"]["upstream_solver_closure"]
]
EXPECTED.extend(
    {"name": name, "version": "0.9.0", "source": GIT}
    for name in ("poulpy-ckks", "poulpy-core", "poulpy-hal", "poulpy-cpu-portable")
)


def dependency_mismatches(packages):
    """Compare package identities only; never claim resolved features or bounds."""
    mismatches = []
    for wanted in EXPECTED:
        found = [package for package in packages if package["name"] == wanted["name"]]
        if len(found) != 1 or any(found[0].get(key) != value for key, value in wanted.items()):
            mismatches.append({"expected": wanted, "actual": found})
    return mismatches


class FrozenDependencyTests(unittest.TestCase):
    def test_consumer_lock_matches_frozen_solver_and_poulpy_identities(self):
        with (HERE / "Cargo.lock").open("rb") as stream:
            lock = tomllib.load(stream)
        self.assertEqual(
            dependency_mismatches(lock["package"]),
            [],
            "Consumer dependency drift: restore frozen version/checksum/source; "
            "do not rewrite source manifests",
        )

    def test_drift_missing_duplicate_and_wrong_source_fail_closed(self):
        packages = copy.deepcopy(EXPECTED)
        self.assertEqual(dependency_mismatches(packages), [])
        for index, package in enumerate(packages):
            for field, value in (
                ("version", "0.0.0"),
                ("checksum", "0" * 64),
                ("source", "path+local"),
            ):
                if field not in package:
                    continue
                with self.subTest(package=package["name"], field=field):
                    drifted = copy.deepcopy(packages)
                    drifted[index][field] = value
                    self.assertTrue(dependency_mismatches(drifted))
                    del drifted[index][field]
                    self.assertTrue(dependency_mismatches(drifted))
            with self.subTest(package=package["name"], case="missing"):
                self.assertTrue(dependency_mismatches(packages[:index] + packages[index + 1 :]))
            with self.subTest(package=package["name"], case="duplicate"):
                self.assertTrue(dependency_mismatches(packages + [package]))

    def test_unrelated_packages_do_not_claim_full_graph_validation(self):
        packages = copy.deepcopy(EXPECTED)
        packages.append({"name": "unrelated", "version": "0.0.0", "source": "path+local"})
        self.assertEqual(dependency_mismatches(packages), [])


if __name__ == "__main__":
    unittest.main()
