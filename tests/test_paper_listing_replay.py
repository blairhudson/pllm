"""Listing provenance, rejection and cleanup tests; no checkpoint or model execution."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import venv


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load("listing_driver", ROOT / "scripts/replay_paper_listing.py")
verifier = load("listing_verifier", ROOT / "paper/artifacts/listing1/verify.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pllm-listing-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "run"
        self.model_file = self.root / "config.json"
        self.model_file.write_text('{"fixture": 1}')
        self.spec = json.loads((ROOT / "paper/artifacts/listing1/qwen-v1/experiment.json").read_text())
        self.model = SimpleNamespace(**self.spec["pipeline"]["model"])
        self.experiment = SimpleNamespace(pipeline=SimpleNamespace(model=self.model), to_spec=lambda: self.spec)
        self.identity = {"schema": "pllm.paper_listing_package.v1", "python_executable": sys.executable,
                         "python_sha256": sha(b"python"), "isolated": 1, "no_user_site": 1,
                         "user_site_enabled": False, "python_environment_scrubbed": True,
                         "site_packages": ["/fixture/site-packages"],
                         "package_file": "/fixture/site-packages/pllm/__init__.py",
                         "native_file": "/fixture/site-packages/pllm/_native.abi3.so",
                         "native_sha256": sha(b"native"), "sdk_sources": {"__init__.py": sha(b"sdk")}}
        attributes = {"sdk_sources": self.identity["sdk_sources"], "native_sha256": self.identity["native_sha256"],
                      "python_binary": self.identity["python_sha256"]}
        self.build = {"attributes": attributes, "digest": sha(canonical(attributes))}
        self.usage = {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48}
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.client.responses.create.return_value = SimpleNamespace(
            model=self.model.source, output_text="fixture text", usage=SimpleNamespace(to_dict=lambda: self.usage))
        self.client.privacy_audit.to_dict.return_value = {
            "online_steps": 4512, "plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0}
        self.gateway = MagicMock()
        self.gateway.__enter__.return_value = self.gateway
        self.reply = self.gateway.post.return_value
        self.reply.status_code = 200
        self.reply.json.return_value = {"usage": self.usage, "output": [{"content": [{"text": "fixture text"}]}]}
        self.roles = MagicMock()
        self.roles.client.return_value = self.client
        self.processes = []
        self.original_launcher = self.spawn
        self.servers = SimpleNamespace(_spawn_role_process=self.original_launcher,
                                       build_roles=lambda *a, **k: self.roles)
        self.watchdog = MagicMock(error=None, maximum_swap_growth_bytes=0)
        self.rt = SimpleNamespace(inspect=lambda: self.build, resolve_model=lambda _: self.resolve(),
            memory=SimpleNamespace(benchmark_memory=lambda *a, **k: {"admitted": True, "selected_backend": "native"},
                                   require_admission=lambda _: None, MemoryWatchdog=lambda *a: self.watchdog),
            servers=self.servers, TestClient=lambda _: self.gateway)
        self.roles.__enter__.side_effect = self.start_roles
        self.roles.close.side_effect = self.close_roles
        self.roles.__exit__.side_effect = lambda *a: self.close_roles()

    def resolve(self):
        rows = [{"path": "config.json", "size": self.model_file.stat().st_size,
                 "sha256": sha(self.model_file.read_bytes())}]
        checkpoint = sha(b"pllm.checkpoint.v1\0" + canonical({"files": rows}))
        lock = {"schema": "pllm.model_source_lock.v1", "kind": "huggingface", "repo_id": self.model.source,
                "revision": self.model.revision, "commit": self.model.revision,
                "files": rows, "checkpoint_digest": checkpoint}
        return SimpleNamespace(manifest=SimpleNamespace(metadata={"source_lock": lock}),
                               checkpoint_digest=checkpoint,
                               source_lock_digest=sha(b"pllm.model_source_lock.v1\0" + canonical(lock)))

    def spawn(self, command, *, environment, **kwargs):
        self.assertEqual(command[1:3], ["-I", "-B"])
        self.assertNotIn("PYTHONPATH", environment)
        process = MagicMock(pid=10001 + len(self.processes))
        process.poll.return_value = None
        self.processes.append(process)
        driver.write(Path(command[5]), {"schema": "pllm.paper_listing_child.v1", "pid": process.pid,
            "parent_pid": os.getpid(), "role": command[10], "status": "identified",
            "phase": "before_module_execution", "identity": self.identity,
            "driver_sha256": driver.digest(command[3]), "expected_parent_sha256": driver.digest(command[6])})
        return process

    def start_roles(self):
        for role in ("inference", "preparation"):
            self.servers._spawn_role_process([sys.executable, "-m", "pllm", "serve", role], environment=dict(os.environ))
        return self.roles

    def close_roles(self):
        for process in self.processes:
            process.poll.return_value = -15

    def run_replay(self, **kwargs):
        with patch.dict(os.environ, {"PYTHONPATH": "poison", "PYTHONHOME": "poison", "PYTHONUSERBASE": "poison"}), \
             patch.object(driver, "runtime", return_value=self.rt), \
             patch.object(driver, "package_identity", return_value=copy.deepcopy(self.identity)), \
             patch.dict(driver.__dict__, {"exec": lambda code, ns: ns.update(experiment=self.experiment)}):
            result = driver.replay(self.output, **kwargs)
        self.assertIs(self.servers._spawn_role_process, self.original_launcher)
        return result

    def test_success_records_child_pids_source_boundaries_and_all_hashes(self):
        result = self.run_replay()
        self.assertEqual(result["status"], "completed")
        self.assertEqual({row["pid"] for row in result["provider_children"]}, {10001, 10002})
        self.assertTrue(result["source_unchanged"])
        verifier.verify_run(self.output, "completed")
        verifier.verify_current_listing(self.output)
        self.watchdog.close.assert_called_once()
        self.assertTrue(all(row["returncode"] == -15 for row in result["provider_children"]))
        with self.assertRaises(FileExistsError):
            self.run_replay()  # A rerun cannot overwrite any earlier attempt.

    def test_admission_only_has_locks_but_never_starts_roles(self):
        result = self.run_replay(preflight_only=True)
        self.assertEqual(result["status"], "admitted_not_executed")
        self.roles.__enter__.assert_not_called()
        self.watchdog.start.assert_not_called()
        verifier.verify_run(self.output, "admitted_not_executed")

    def test_gateway_failure_retains_status_not_response_body(self):
        self.reply.status_code = 503
        self.reply.json.return_value = {"private": "credential-and-private-text"}
        result = self.run_replay()
        self.assertEqual((result["failure_phase"], result["failure_reason"], result["gateway_status"]),
                         ("gateway_request", "gateway_http_status", 503))
        self.assertEqual(result["sdk_usage"], self.usage)
        self.reply.json.assert_not_called()
        self.assertNotIn("credential-and-private-text", (self.output / "result.json").read_text())
        self.roles.close.assert_called()
        self.watchdog.close.assert_called_once()
        verifier.verify_run(self.output, "rejected")

    def test_primary_failure_survives_cleanup_failure_without_private_exception_text(self):
        self.client.responses.create.side_effect = RuntimeError("private prompt and credential")
        self.client.__exit__.side_effect = OSError("secret cleanup text")
        result = self.run_replay()
        self.assertEqual(result["failure_phase"], "sdk_request")
        self.assertEqual(result["failure_reason"], "operation_failed")
        self.assertEqual(result["secondary_failures"][0]["phase"], "sdk_cleanup")
        raw = (self.output / "result.json").read_text()
        self.assertNotIn("private prompt and credential", raw)
        self.assertNotIn("secret cleanup text", raw)
        self.assertTrue(all(item["returncode"] is not None for item in result["provider_children"]))
        self.watchdog.close.assert_called_once()

    def test_partial_startup_is_reaped_and_watchdog_stops(self):
        def fail_start():
            self.servers._spawn_role_process([sys.executable, "-m", "pllm", "serve", "inference"], environment={})
            raise RuntimeError("private startup detail")
        self.roles.__enter__.side_effect = fail_start
        result = self.run_replay()
        self.assertEqual(result["failure_phase"], "provider_startup")
        self.assertEqual(result["provider_children"][0]["returncode"], -15)
        self.watchdog.close.assert_called_once()

    def test_watchdog_rejection_precedes_provider_startup(self):
        self.watchdog.start.side_effect = RuntimeError("private host detail")
        result = self.run_replay()
        self.assertEqual(result["failure_phase"], "watchdog_start")
        self.roles.__enter__.assert_not_called()
        self.roles.close.assert_called()
        self.watchdog.close.assert_called_once()

    def test_changed_source_rejects_and_preserves_both_inventories(self):
        def mutate(*args, **kwargs):
            self.model_file.write_text('{"fixture": 2}')
            return self.reply
        self.gateway.post.side_effect = mutate
        result = self.run_replay()
        self.assertEqual((result["failure_phase"], result["failure_reason"]), ("source_after", "source_lock_changed"))
        self.assertFalse(result["source_unchanged"])
        self.assertNotEqual((self.output / "source-lock-before.json").read_bytes(),
                            (self.output / "source-lock-after.json").read_bytes())
        verifier.verify_run(self.output, "rejected")

    def test_source_lock_rejects_wrong_revision_and_stale_file_digest(self):
        for field, value, reason in (("commit", "0" * 40, "source_revision_mismatch"),
                                     ("checkpoint_digest", "0" * 64, "source_digest_mismatch")):
            with self.subTest(field=field):
                source = self.resolve()
                source.manifest.metadata["source_lock"][field] = value
                with self.assertRaisesRegex(driver.ReplayFailure, reason):
                    driver.source_lock(source, self.model)
        source = self.resolve()
        source.manifest.metadata["source_lock"]["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(driver.ReplayFailure, "source_digest_mismatch"):
            driver.source_lock(source, self.model)

    def test_verifier_rejects_semantic_tamper_even_with_updated_file_hash(self):
        result = self.run_replay()
        path = self.output / "provider-inference.json"
        receipt = json.loads(path.read_text())
        receipt["pid"] += 1
        driver.write(path, receipt)
        result["files"][path.name] = driver.digest(path)
        driver.write(self.output / "result.json", result)
        with self.assertRaisesRegex(ValueError, "Child process binding mismatch"):
            verifier.verify_run(self.output, "completed")

    def test_verifier_checks_source_revision_and_content_digest_beyond_file_hashes(self):
        result = self.run_replay()
        original = json.loads((self.output / "source-lock-before.json").read_text())
        for field, message in (("revision", "Source revision mismatch"), ("file", "Checkpoint digest")):
            with self.subTest(field=field):
                lock = copy.deepcopy(original)
                if field == "revision":
                    lock["commit"] = "0" * 40
                else:
                    lock["files"][0]["sha256"] = "0" * 64
                for name in ("source-lock-before.json", "source-lock-after.json"):
                    driver.write(self.output / name, lock)
                    result["files"][name] = driver.digest(self.output / name)
                driver.write(self.output / "result.json", result)
                with self.assertRaisesRegex(ValueError, message):
                    verifier.verify_run(self.output, "completed")

    def test_original_snapshots_remain_frozen_despite_current_driver_version(self):
        here = ROOT / "paper/artifacts/listing1"
        index = json.loads((here / "index.json").read_text())
        verifier.verify_files(here, index["files"])
        verifier.verify_run(here / "qwen-v1", "completed")
        verifier.verify_run(here / "preflight-v1", "admitted_not_executed")
        verifier.verify_current_listing(here / "qwen-v1")
        self.assertNotEqual(driver.digest(ROOT / "scripts/replay_paper_listing.py"),
                            driver.digest(here / "qwen-v1/driver.py"))


class ArchiveLinkageTests(unittest.TestCase):
    """Read-only linkage tests against disposable archive copies, never live runtimes."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pllm-listing-linkage-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "relocated-checkout"
        self.here = self.root / "paper/artifacts/listing1"
        self.here.mkdir(parents=True)
        original = ROOT / "paper/artifacts/listing1"
        for name in ("index.json", "index-v2.json"):
            shutil.copyfile(original / name, self.here / name)
        for name in ("preflight-v1", "qwen-v1", "qwen-fresh-v2"):
            shutil.copytree(original / name, self.here / name)
        shutil.copyfile(ROOT / "paper/manuscript.md", self.root / "paper/manuscript.md")
        self.index = json.loads((self.here / "index-v2.json").read_text())
        self.fresh = self.here / self.index["archive"]
        for link in self.index["release"].values():
            destination = self.root / link["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / link["path"], destination)
        build = self.root / self.index["release"]["build_receipt"]["path"]
        wheel_name = json.loads(build.read_text())["wheel"]
        wheel_relative = Path(self.index["release"]["build_receipt"]["path"]).parent / wheel_name
        self.wheel = self.root / wheel_relative
        self.wheel.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / wheel_relative, self.wheel)

    def verify(self):
        return verifier.verify_archives(self.here, self.root)

    def save_index(self):
        driver.write(self.here / "index-v2.json", self.index)

    def rewrite_release(self, role, update):
        link = self.index["release"][role]
        path = self.root / link["path"]
        document = json.loads(path.read_text())
        update(document)
        driver.write(path, document)
        link["sha256"] = verifier.file_hash(path)
        self.save_index()

    def rewrite_qwen_identity(self, identity):
        # Make the run internally self-consistent, so release linkage must catch
        # a different native/Python/source identity rather than just a stale hash.
        for name in ("parent-identity.json", "parent-identity-after.json"):
            driver.write(self.fresh / name, identity)
        parent_hash = verifier.file_hash(self.fresh / "parent-identity.json")
        for role in ("inference", "preparation"):
            path = self.fresh / f"provider-{role}.json"
            receipt = json.loads(path.read_text())
            receipt.update(identity=identity, expected_parent_sha256=parent_hash)
            driver.write(path, receipt)
        build = json.loads((self.fresh / "build.json").read_text())
        build["attributes"].update(sdk_sources=identity["sdk_sources"], native_sha256=identity["native_sha256"],
                                   python_binary=identity["python_sha256"])
        build["digest"] = sha(canonical(build["attributes"]))
        driver.write(self.fresh / "build.json", build)
        result = json.loads((self.fresh / "result.json").read_text())
        result["files"] = {name: verifier.file_hash(self.fresh / name) for name in result["files"]}
        driver.write(self.fresh / "result.json", result)
        self.index["files"] = {name: verifier.file_hash(self.here / name) for name in self.index["files"]}
        self.save_index()
        verifier.verify_run(self.fresh, "completed")

    def test_default_verification_is_read_only_and_works_without_live_environment(self):
        def snapshot():
            return {p.relative_to(self.root).as_posix(): verifier.file_hash(p)
                    for p in self.root.rglob("*") if p.is_file()}
        before = snapshot()
        output = io.StringIO()
        with patch.object(verifier, "HERE", self.here), patch.object(verifier, "ROOT", self.root), \
             patch.object(sys, "argv", ["verify.py"]), patch("sys.stdout", output), \
             patch("subprocess.run", side_effect=AssertionError("Verifier executed a subprocess")), \
             patch("subprocess.Popen", side_effect=AssertionError("Verifier launched a provider")):
            verifier.main()
        self.assertIn("12 original + 12 fresh frozen files", output.getvalue())
        self.assertIn("4 linked release metadata files", output.getvalue())
        self.assertIn("Qwen-observed 313 SDK sources", output.getvalue())
        self.assertIn("355-file full package inventory belongs to earlier installation evidence", output.getvalue())
        self.assertEqual(before, snapshot())

    def test_internally_consistent_qwen_identity_must_match_release(self):
        original = json.loads((self.fresh / "parent-identity.json").read_text())
        cases = (("native_sha256", "Qwen/release native mismatch"),
                 ("python_sha256", "Qwen/release Python hash mismatch"),
                 ("sdk_sources", "Qwen/release SDK inventory mismatch"))
        for field, message in cases:
            with self.subTest(field=field):
                identity = copy.deepcopy(original)
                if field == "sdk_sources":
                    identity[field]["__init__.py"] = "0" * 64
                else:
                    identity[field] = "0" * 64
                self.rewrite_qwen_identity(identity)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()

    def test_package_inventory_is_bound_to_wheel_not_only_native(self):
        def tamper(provenance):
            provenance["package_files"]["pllm/__init__.pyi"] = "0" * 64
            provenance["package_digest"] = sha(canonical(provenance["package_files"]))
        self.rewrite_release("installed_provenance", tamper)
        with self.assertRaisesRegex(ValueError, "Fresh release package inventory/digest"):
            self.verify()

    def test_wheel_bytes_are_checked_without_running_or_installing_them(self):
        with self.wheel.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "Linked wheel identity"):
            self.verify()

    def test_linked_metadata_hashes_cannot_drift(self):
        path = self.root / self.index["release"]["installed_provenance"]["path"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Linked metadata hash changed: installed_provenance"):
            self.verify()

    def test_original_blocked_build_cannot_be_relabelled_passed(self):
        self.rewrite_release("build_receipt", lambda build: build.update(status="passed"))
        with self.assertRaisesRegex(ValueError, "Original build history relabelled"):
            self.verify()

    def test_continuation_must_bind_original_manifest(self):
        self.rewrite_release("environment_receipt", lambda receipt: receipt.update(build_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Continuation build manifest binding"):
            self.verify()

    def test_fresh_archive_index_must_cover_result_itself(self):
        del self.index["files"]["qwen-fresh-v2/result.json"]
        self.save_index()
        with self.assertRaisesRegex(ValueError, "Fresh archive index coverage"):
            self.verify()

    def test_original_index_bytes_are_retained(self):
        path = self.here / "index.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Original index changed"):
            self.verify()

    def test_operational_directories_cannot_enter_publication_archive(self):
        (self.fresh / "role-logs").mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected operational archive entries"):
            self.verify()

    def test_current_listing_is_verified_separately(self):
        manuscript = self.root / "paper/manuscript.md"
        text = manuscript.read_text()
        old = '"prepared-qwen"'
        self.assertIn(old, text)
        manuscript.write_text(text.replace(old, "'prepared-qwen'"))
        with self.assertRaisesRegex(ValueError, "Published listing is not unique"):
            self.verify()

    def test_link_paths_are_repo_relative_and_bounded(self):
        self.index["release"]["installed_provenance"]["path"] = "../outside.json"
        self.save_index()
        with self.assertRaisesRegex(ValueError, "Linked path escapes its root"):
            self.verify()

    def test_accepted_command_must_name_observed_interpreter(self):
        self.index["accepted_invocation"][0] = "target/frontier-sdk-env/bin/python"
        self.save_index()
        with self.assertRaisesRegex(ValueError, "Accepted invocation binding"):
            self.verify()


class SubprocessIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pllm-listing-isolation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        env = self.root / "env"
        # A tiny synthetic installed package, no pip, native build or model.
        venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
        self.python = str(env / "bin/python")
        clean = driver.isolated_environment(os.environ)
        purelib = subprocess.check_output([self.python, "-I", "-B", "-c",
                                          "import sysconfig; print(sysconfig.get_path('purelib'))"],
                                         env=clean, text=True).strip()
        package = Path(purelib) / "pllm"
        package.mkdir()
        (package / "__init__.py").write_text('identity = "installed fixture"\n')
        (package / "_native.py").write_text('identity = "synthetic native fixture"\n')
        (package / "__main__.py").write_text(
            "import json, os, sys\nfrom pathlib import Path\nimport pllm, pllm._native\n"
            "print(json.dumps({'pid': os.getpid(), 'isolated': sys.flags.isolated, "
            "'package': pllm.__file__, 'native': pllm._native.__file__, "
            "'receipt_before_main': Path(os.environ['TEST_RECEIPT']).is_file(), "
            "'python_environment': sorted(k for k in os.environ if k in "
            "('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'))}))\n")
        self.output = self.root / "run"
        self.output.mkdir()
        shutil.copyfile(ROOT / "scripts/replay_paper_listing.py", self.output / "driver.py")
        code = ("import json, runpy; ns=runpy.run_path(" + repr(str(self.output / "driver.py"))
                + "); print(json.dumps(ns['package_identity']()))")
        self.identity = json.loads(subprocess.check_output([self.python, "-I", "-B", "-c", code],
                                                           env=clean, text=True))
        driver.write(self.output / "parent-identity.json", self.identity)
        self.poison = self.root / "poison"
        self.poison.mkdir()
        (self.poison / "pllm").mkdir()
        (self.poison / "pllm/__init__.py").write_text('raise RuntimeError("cwd/path import poison")\n')
        (self.poison / "sitecustomize.py").write_text('raise RuntimeError("site import poison")\n')
        self.environment = dict(os.environ, PYTHONPATH=str(self.poison), PYTHONHOME=str(self.poison),
                                PYTHONUSERBASE=str(self.poison))
        def spawn(command, *, environment, **kwargs):
            return subprocess.Popen(command, cwd=self.poison, env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.original = spawn
        self.servers = SimpleNamespace(_spawn_role_process=spawn)

    def test_actual_children_ignore_cwd_path_home_and_user_site_and_report_loaded_modules(self):
        children = []
        with driver.isolated_providers(self.servers, self.output, self.identity, children):
            for role in ("inference", "preparation"):
                receipt_path = self.output / f"provider-{role}.json"
                environment = dict(self.environment, TEST_RECEIPT=str(receipt_path))
                process = self.servers._spawn_role_process(
                    [self.python, "-m", "pllm", "serve", role], environment=environment)
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stderr)
                actual = json.loads(stdout)
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(actual["pid"], process.pid)
                self.assertEqual(receipt["pid"], process.pid)
                self.assertEqual(str(Path(actual["package"]).resolve()), self.identity["package_file"])
                self.assertEqual(str(Path(actual["native"]).resolve()), self.identity["native_file"])
                self.assertEqual(actual["isolated"], 1)
                self.assertEqual(actual["python_environment"], [])
                self.assertTrue(actual["receipt_before_main"])
                self.assertEqual(receipt["identity"], self.identity)
            driver.check_children(self.output, self.identity, children)
        self.assertIs(self.servers._spawn_role_process, self.original)

    def test_mismatched_actual_native_identity_rejects_before_provider_module(self):
        expected = copy.deepcopy(self.identity)
        expected["native_sha256"] = "0" * 64
        driver.write(self.output / "parent-identity.json", expected)
        receipt_path = self.output / "provider-inference.json"
        with driver.isolated_providers(self.servers, self.output, expected, []):
            process = self.servers._spawn_role_process([self.python, "-m", "pllm", "serve", "inference"],
                environment=dict(self.environment, TEST_RECEIPT=str(receipt_path)))
            stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 1, stderr)
        self.assertEqual(stdout, "")
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["failure"]["reason"], "provider_identity_mismatch")
        self.assertEqual(receipt["identity"], self.identity)
        self.assertEqual(receipt["pid"], process.pid)

    def test_parent_cli_rejects_nonisolated_python_before_any_model_resolution(self):
        output = self.root / "nonisolated"
        completed = subprocess.run([self.python, "-B", str(ROOT / "scripts/replay_paper_listing.py"),
                                    "--output", str(output)], cwd=self.poison,
                                   env=driver.isolated_environment(os.environ),
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stderr.strip(), "Replay rejected: parent_identity/isolated_python_required")
        result = verifier.verify_run(output, "rejected")
        self.assertEqual(result["provider_children"], [])
        self.assertEqual(set(result["files"]), {"driver.py"})


if __name__ == "__main__":
    unittest.main()
