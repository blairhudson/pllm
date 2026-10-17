"""Destructive retention gates: evidence, active owners, pins and recovery."""

from contextlib import contextmanager
from pathlib import Path
import os
import time
from types import SimpleNamespace
import zipfile

import pytest

from scripts import maintain_target as maintenance


def cache(target, name):
    base = target / "debug"
    base.mkdir(parents=True, exist_ok=True)
    (base / ".cargo-lock").touch()
    path = base / "incremental" / name
    path.mkdir(parents=True)
    (path / "s-session").mkdir()
    (path / "s-session/query-cache.bin").write_bytes(b"cache")
    return path


def fixture(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    paths = [cache(target, f"pllm_core-{index}") for index in range(4)]
    return target, paths


def future_plan(target, **kwargs):
    return maintenance.plan(target, now=time.time() + 40 * maintenance.DAY,
                            ownership=([], []), **kwargs)


def test_dry_run_and_apply_keep_newest_and_evidence(tmp_path):
    target, paths = fixture(tmp_path)
    evidence = target / "research/old/report.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("unique evidence")
    report = future_plan(target)
    assert len(report["selected"]) == 2
    assert all(path.exists() for path in paths)
    result = maintenance.apply(report, ownership=([], []))
    assert len(result["removed"]) == 2
    assert sum(path.exists() for path in paths) == 2
    assert evidence.read_text() == "unique evidence"
    assert Path(result["plan_receipt"]).is_file()


def test_recent_files_override_old_parent(tmp_path):
    target, _ = fixture(tmp_path)
    assert maintenance.plan(target, ownership=([], []))["selected"] == []


def test_pins_at_ancestor_and_inside_tree(tmp_path):
    target, paths = fixture(tmp_path)
    (paths[0] / "s-session" / maintenance.PIN).touch()
    (paths[1] / maintenance.PIN).touch()
    assert future_plan(target)["selected"] == []
    (target / maintenance.PIN).touch()
    assert future_plan(target)["selected"] == []


def test_active_owner_and_compiler_are_retained(tmp_path):
    target, paths = fixture(tmp_path)
    report = maintenance.plan(target, now=time.time() + 40 * maintenance.DAY,
                              ownership=([paths[0] / "s-session/query-cache.bin"], []))
    assert all(row["path"] != str(paths[0].relative_to(target)) for row in report["selected"])
    report = future_plan(target)
    result = maintenance.apply(report, ownership=([], [123]))
    assert result["removed"] == []
    assert all(path.exists() for path in paths)


def test_changed_or_newly_pinned_candidate_survives(tmp_path):
    target, paths = fixture(tmp_path)
    report = future_plan(target)
    (paths[0] / "s-session/query-cache.bin").write_text("changed")
    (paths[1] / maintenance.PIN).touch()
    assert maintenance.apply(report, ownership=([], []))["removed"] == []


def test_links_and_external_paths_are_never_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important").write_text("retain")
    target, _ = fixture(tmp_path)
    (target / "linked-venv").symlink_to(outside)
    report = future_plan(target)
    victim = target / report["selected"][0]["path"]
    victim.rename(victim.with_name("retained-original"))
    victim.symlink_to(outside)
    maintenance.apply(report, ownership=([], []))
    assert (outside / "important").read_text() == "retain"
    link = tmp_path / "target-link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="real directory"):
        future_plan(link)


def make_venv(target, name, body=b"native"):
    path = target / name
    site = path / "lib/python3.13/site-packages"
    package = site / "pllm"
    package.mkdir(parents=True)
    (package / "_native.abi3.so").write_bytes(body)
    (package / "__init__.py").write_text("# installed")
    (path / "pyvenv.cfg").write_text("version = 3.13.9\n")
    metadata = site / "pllm_run-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: pllm.run\nVersion: 0.1\n")
    return path


def make_wheel(target, body=b"native"):
    path = target / "wheels/pllm_run-0.1-py3-none-any.whl"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("pllm/_native.abi3.so", body)
        wheel.writestr("pllm/__init__.py", "# installed")
    return path


def test_venv_requires_complete_matching_wheel_and_preserves_receipt(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    old = make_venv(target, "old")
    for index in range(3):
        make_venv(target, f"recent{index}", b"different")
    assert future_plan(target)["selected"] == []
    wheel = make_wheel(target)
    report = future_plan(target)
    assert len(report["selected"]) == 1
    assert report["selected"][0]["recreate"]["requirements"] == ["pllm.run==0.1"]
    result = maintenance.apply(report, ownership=([], []))
    assert len(result["removed"]) == 1
    assert not old.exists() and wheel.exists()
    assert "package_sha256" in Path(result["plan_receipt"]).read_text()


def test_venv_modified_python_or_extra_root_files_are_preserved(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    old = make_venv(target, "old")
    make_wheel(target)
    package = old / "lib/python3.13/site-packages/pllm/__init__.py"
    package.write_text("# unique source")
    for index in range(3):
        make_venv(target, f"recent{index}")
    assert future_plan(target)["selected"] == []
    (old / "report.json").write_text("retain evidence")
    package.write_text("# installed")
    assert old not in [path for path, _, _ in maintenance.discover(target)[0]]


def test_changed_wheel_blocks_removal(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    old = make_venv(target, "old")
    for index in range(3):
        make_venv(target, f"recent{index}")
    wheel = make_wheel(target)
    report = future_plan(target)
    wheel.write_bytes(b"changed")
    assert maintenance.apply(report, ownership=([], []))["removed"] == []
    assert old.exists()


def test_busy_cargo_lock_blocks_removal(tmp_path):
    target, paths = fixture(tmp_path)
    report = future_plan(target)
    with maintenance.locked(target / "debug/.cargo-lock"):
        assert maintenance.apply(report, ownership=([], []))["removed"] == []
    assert all(path.exists() for path in paths)


@pytest.mark.parametrize("kind", ["incremental", "venv", "pin"])
def test_candidate_refreshed_during_lock_acquisition_survives(tmp_path, monkeypatch, kind):
    if kind == "venv":
        target = tmp_path / "target"
        target.mkdir()
        candidate = make_venv(target, "old")
        for index in range(3):
            make_venv(target, f"recent{index}")
        make_wheel(target)
    else:
        target, paths = fixture(tmp_path)
        candidate = paths[0]
    report = future_plan(target)
    original = maintenance.locked

    @contextmanager
    def refreshed(path):
        with original(path):
            if path.name in {".cargo-lock", ".lock"}:
                if kind == "pin":
                    (candidate / maintenance.PIN).touch()
                elif kind == "venv":
                    (candidate / "lib/updated.txt").write_text("new work")
                else:
                    (candidate / "s-session/query-cache.bin").write_bytes(b"fresh cache")
            yield

    monkeypatch.setattr(maintenance, "locked", refreshed)
    result = maintenance.apply(report, ownership=([], []))
    assert candidate.exists()
    assert any(row["path"] == str(candidate.relative_to(target))
               and row["reason"] == "candidate changed while acquiring lock"
               for row in result["skipped"])


def test_pressure_never_removes_recent_or_pinned_work(tmp_path):
    target, _ = fixture(tmp_path)
    report = maintenance.plan(target, budget_bytes=1, free_floor_bytes=1 << 60, ownership=([], []))
    assert report["pressure"] and report["selected"] == []


def test_cleanup_lock_and_receipt_failure_preserve_candidates(tmp_path, monkeypatch):
    target, paths = fixture(tmp_path)
    report = future_plan(target)
    state = target / ".maintenance"
    state.mkdir()
    with maintenance.locked(state / "cleanup.lock"), pytest.raises(BlockingIOError):
        maintenance.apply(report, ownership=([], []))
    def fail(*args):
        raise OSError("full disk")
    monkeypatch.setattr(maintenance, "write_json", fail)
    with pytest.raises(OSError, match="full disk"):
        maintenance.apply(report, ownership=([], []))
    assert all(path.exists() for path in paths)


def test_process_ownership_keeps_paths_and_rejects_unknown_developer(tmp_path, monkeypatch):
    class Process:
        pid = -1
        executable = "/usr/libexec/lsd"

        def uids(self):
            return SimpleNamespace(effective=os.getuid())

        def status(self):
            return "running"

        def cmdline(self):
            return ["target/old/bin/python", "script.py"]

        def cwd(self):
            return str(tmp_path)

        def open_files(self):
            raise maintenance.psutil.AccessDenied(self.pid)

        def name(self):
            return Path(self.executable).name

        def exe(self):
            return self.executable

    process = Process()
    monkeypatch.setattr(maintenance.psutil, "process_iter", lambda: [process])
    paths, _ = maintenance.active_paths()
    assert tmp_path / "target/old" in maintenance.owner_ancestors(paths)
    process.executable = "/opt/python"
    with pytest.raises(RuntimeError, match="ownership"):
        maintenance.active_paths()


def test_ancestor_symlink_swap_cannot_delete_external_tree(tmp_path):
    target, paths = fixture(tmp_path)
    identity = [paths[0].stat().st_dev, paths[0].stat().st_ino]
    external = tmp_path / "external"
    external.mkdir()
    important = external / paths[0].name
    important.mkdir()
    (important / "report.json").write_text("preserve")
    incremental = target / "debug/incremental"
    incremental.rename(target / "debug/original")
    incremental.symlink_to(external)
    with pytest.raises(OSError):
        maintenance.remove_tree(target, paths[0], identity)
    assert (important / "report.json").read_text() == "preserve"


def test_daily_job_is_scoped_to_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    calls = []
    monkeypatch.setattr(maintenance.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    result = maintenance.install_daily(tmp_path / "repo")
    definition = maintenance.plistlib.loads(Path(result["plist"]).read_bytes())
    assert definition["StartInterval"] == 86400
    assert definition["ProgramArguments"] == [str(tmp_path / "repo/.venv/bin/python"), "-B",
                                               str(tmp_path / "repo/scripts/maintain_target.py"), "--apply"]
    assert calls[-1] == ["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", result["plist"]]
