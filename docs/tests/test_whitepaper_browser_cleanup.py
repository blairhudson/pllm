"""Bounded shutdown of the exact browser owned by the whitepaper build."""

from __future__ import annotations

import importlib.util
import io
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class WhitepaperBrowserCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location("paper_builder_cleanup", ROOT / "docs/scripts/build-papers.py")
        self.builder = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {spec.name: self.builder}):
            spec.loader.exec_module(self.builder)
        self.process = mock.Mock(pid=12345)
        self.process.poll.return_value = None
        self.profile = ROOT / "target/pllm-whitepaper-test/profile"
        self.profile_probe = self.builder.whitepaper_profile_processes
        self.helpers = self.enterContext(mock.patch.object(self.builder, "whitepaper_profile_processes",
                                                          return_value=[]))

    def timeout(self) -> subprocess.TimeoutExpired:
        return subprocess.TimeoutExpired("owned-browser", 5)

    def test_natural_exit_is_reaped_before_any_termination_signal(self) -> None:
        with mock.patch.object(self.builder.os, "killpg", side_effect=ProcessLookupError()) as killpg:
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.process.wait.assert_called_once_with(timeout=2)
        self.process.send_signal.assert_not_called()
        killpg.assert_called_once_with(self.process.pid, 0)

    def test_group_disappearing_between_poll_and_signal_is_a_safe_race(self) -> None:
        self.process.wait.side_effect = [self.timeout(), None]
        with mock.patch.object(self.builder.os, "killpg", side_effect=ProcessLookupError()) as killpg:
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.assertEqual(killpg.call_args_list,
                         [mock.call(self.process.pid, signal.SIGTERM), mock.call(self.process.pid, 0)])
        self.assertEqual(self.process.wait.call_args_list, [mock.call(timeout=2), mock.call(timeout=5)])

    def test_group_permission_denial_falls_back_only_to_owned_leader(self) -> None:
        self.process.wait.side_effect = [self.timeout(), None]
        with mock.patch.object(self.builder.os, "killpg",
                               side_effect=[PermissionError(), ProcessLookupError()]) as killpg:
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.process.send_signal.assert_called_once_with(signal.SIGTERM)
        self.assertEqual(killpg.call_args_list,
                         [mock.call(self.process.pid, signal.SIGTERM), mock.call(self.process.pid, 0)])

    def test_permission_fallback_handles_leader_exit_race(self) -> None:
        self.process.wait.side_effect = [self.timeout(), None]
        self.process.send_signal.side_effect = ProcessLookupError()
        with mock.patch.object(self.builder.os, "killpg",
                               side_effect=[PermissionError(), ProcessLookupError()]):
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.assertEqual(self.process.wait.call_args_list, [mock.call(timeout=2), mock.call(timeout=5)])

    def test_term_timeout_escalates_with_bounded_kill_wait(self) -> None:
        self.process.wait.side_effect = [self.timeout(), self.timeout(), None]
        with mock.patch.object(self.builder.os, "killpg",
                               side_effect=[None, None, ProcessLookupError()]) as killpg:
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.assertEqual(killpg.call_args_list, [mock.call(self.process.pid, signal.SIGTERM),
                                                mock.call(self.process.pid, signal.SIGKILL),
                                                mock.call(self.process.pid, 0)])
        self.assertEqual(self.process.wait.call_args_list,
                         [mock.call(timeout=2), mock.call(timeout=5), mock.call(timeout=5)])

    def test_unreapable_leader_is_not_reported_as_cleaned_up(self) -> None:
        self.process.wait.side_effect = self.timeout()
        with (mock.patch.object(self.builder.os, "killpg", side_effect=PermissionError()),
              self.assertRaisesRegex(RuntimeError, "Timed out reaping.*cleanup unverified")):
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.assertEqual(self.process.send_signal.call_args_list,
                         [mock.call(signal.SIGTERM), mock.call(signal.SIGKILL)])
        self.assertTrue(all(call.kwargs["timeout"] > 0 for call in self.process.wait.call_args_list))

    def test_denied_leader_signal_reports_exact_ownership(self) -> None:
        self.process.wait.side_effect = self.timeout()
        self.process.send_signal.side_effect = PermissionError("leader denied")
        with (mock.patch.object(self.builder.os, "killpg", side_effect=PermissionError()),
              self.assertRaisesRegex(RuntimeError, "Cannot signal owned browser PID/group 12345") as error):
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        self.assertIn(str(self.profile), str(error.exception))
        self.assertIsInstance(error.exception.__cause__, PermissionError)

    def test_leader_exit_does_not_prove_helper_group_drain(self) -> None:
        for failure, message in ((None, "group members remain"), (PermissionError(), "group visibility denied")):
            with (self.subTest(message=message),
                  mock.patch.object(self.builder.os, "killpg", side_effect=failure) as killpg,
                  mock.patch.object(self.builder.time, "monotonic", side_effect=[0, 6]),
                  self.assertRaisesRegex(RuntimeError, message + "; helper cleanup unverified")):
                self.builder.cleanup_whitepaper_browser(self.process, self.profile)
            killpg.assert_called_once_with(self.process.pid, 0)

    def test_helper_drain_is_bounded_and_can_finish_after_leader_exit(self) -> None:
        with (mock.patch.object(self.builder.os, "killpg", side_effect=[None, ProcessLookupError()]),
              mock.patch.object(self.builder.time, "monotonic", side_effect=[0, 1]),
              mock.patch.object(self.builder.time, "sleep") as sleep):
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        sleep.assert_called_once_with(0.1)

    def test_detached_profile_helper_prevents_false_cleanup_success(self) -> None:
        self.helpers.return_value = [67890]
        with (mock.patch.object(self.builder.os, "killpg", side_effect=ProcessLookupError()) as killpg,
              mock.patch.object(self.builder.time, "monotonic", side_effect=[0, 6]),
              self.assertRaisesRegex(RuntimeError, "profile-associated helpers remain:.*67890")):
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)
        killpg.assert_called_once_with(self.process.pid, 0)
        self.process.send_signal.assert_not_called()

    def test_unavailable_helper_inspection_remains_a_cleanup_failure(self) -> None:
        self.helpers.side_effect = subprocess.TimeoutExpired("ps", 1)
        with (mock.patch.object(self.builder.os, "killpg", side_effect=ProcessLookupError()),
              self.assertRaisesRegex(RuntimeError, "Cannot inspect helpers.*cleanup unverified")):
            self.builder.cleanup_whitepaper_browser(self.process, self.profile)

    def test_profile_inspection_is_read_only_exact_and_bounded(self) -> None:
        with (mock.patch.object(self.builder.subprocess, "run", return_value=mock.Mock(stdout=(
                f"123 browser --user-data-dir={self.profile}\n"
                f"456 crashpad --database={self.profile}/Crashpad\n"
                f"789 browser --user-data-dir={self.profile}-other\n"
                "987 browser --user-data-dir=/unrelated/profile\n"))) as run):
            self.assertEqual(self.profile_probe(self.profile), [123, 456])
        run.assert_called_once_with(["ps", "-axo", "pid=,command="], capture_output=True,
                                    text=True, check=True, timeout=1)

    def test_render_error_stays_primary_when_cleanup_also_fails(self) -> None:
        self.process.poll.return_value = 0
        stderr = io.StringIO()
        with (mock.patch.object(self.builder, "run"),
              mock.patch.object(self.builder, "pandoc_args", return_value=["pandoc"]),
              mock.patch.object(self.builder.subprocess, "Popen", return_value=self.process) as popen,
              mock.patch.object(self.builder.time, "monotonic", side_effect=[0, 0]),
              mock.patch.object(self.builder, "cleanup_whitepaper_browser", side_effect=RuntimeError("unverified")),
              mock.patch.object(sys, "stderr", stderr),
              self.assertRaisesRegex(SystemExit, "Browser did not print whitepaper") as error):
            self.builder.build_whitepaper_pdf(self.builder.PAPERS["whitepaper"], "owned-test-browser")
        self.assertIn("Browser cleanup also failed: unverified", error.exception.__notes__)
        self.assertIn("cleanup also failed", stderr.getvalue())
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.args[0][0], "owned-test-browser")

    def test_valid_pdf_does_not_hide_cleanup_failure_or_get_published(self) -> None:
        def browser(command: list[str], **kwargs) -> mock.Mock:
            pdf = Path(next(arg.removeprefix("--print-to-pdf=") for arg in command
                            if arg.startswith("--print-to-pdf=")))
            pdf.write_bytes(b"PDF fixture\n%%EOF\n")
            return self.process

        with (mock.patch.object(self.builder, "run"),
              mock.patch.object(self.builder, "pandoc_args", return_value=["pandoc"]),
              mock.patch.object(self.builder.subprocess, "Popen", side_effect=browser),
              mock.patch.object(self.builder.time, "monotonic", side_effect=[0, 0]),
              mock.patch.object(self.builder, "cleanup_whitepaper_browser", side_effect=RuntimeError("unverified")),
              mock.patch.object(self.builder, "validate_pdf") as validate,
              mock.patch.object(self.builder.shutil, "copyfile") as copy,
              self.assertRaisesRegex(RuntimeError, "unverified")):
            self.builder.build_whitepaper_pdf(self.builder.PAPERS["whitepaper"], "owned-test-browser")
        validate.assert_not_called()
        copy.assert_not_called()
        self.assertEqual(self.builder.PAPERS["whitepaper"].max_pages, 2)


if __name__ == "__main__":
    unittest.main()
