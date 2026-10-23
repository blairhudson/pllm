"""Fresh-client benchmark workers. One parent-owned private salt crosses the pipe."""
from __future__ import annotations

import multiprocessing
import signal


def _worker(connection, options):
    try:
        from pllm import Experiment
        from pllm.deployment import LinkConditions, WanConditions
        from .benchmark_cli import run_loopback_benchmark

        def terminate(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        experiment = options.pop("experiment_record")
        if experiment is not None:
            options["experiment"] = Experiment.from_spec(experiment[0]).with_params(pipeline__profile=experiment[1])
        for key, cls in (("wan", WanConditions), ("docker_network", LinkConditions)):
            if options.get(key) is not None:
                options[key] = cls.from_spec(options[key])
        options["progress"] = lambda message: connection.send(("progress", message))
        report = run_loopback_benchmark(**options)
        report["configuration"]["client_process_isolated"] = True
        connection.send(("report", report))
    except BaseException as error:
        from .benchmark_failure import sanitize_failure_diagnostics
        diagnostics = sanitize_failure_diagnostics(getattr(error, "failure_diagnostics", None))
        if diagnostics is not None:
            connection.send(("failure_diagnostics", diagnostics))
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def run_isolated_loopback_benchmark(**options):
    from .benchmark_cli import LoopbackBenchmarkError
    progress = options.pop("progress", None)
    experiment = options.pop("experiment", None)
    options["experiment_record"] = None if experiment is None else (experiment.to_spec(), experiment.pipeline.profile)
    for key in ("wan", "docker_network"):
        if options.get(key) is not None:
            options[key] = options[key].to_spec()
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    from .client_process_usage import client_process_factory, finalize_client_report
    process = client_process_factory(context.Process)(target=_worker, args=(send, options))
    process.start()
    send.close()
    diagnostics = None
    report, forced = None, False
    try:
        while True:
            try:
                kind, value = receive.recv()
            except EOFError as error:
                raise LoopbackBenchmarkError("isolated benchmark worker exited without a report") from error
            if kind == "progress" and progress is not None:
                progress(value)
            elif kind == "report":
                report = value
                break
            elif kind == "failure_diagnostics":
                from .benchmark_failure import sanitize_failure_diagnostics
                diagnostics = sanitize_failure_diagnostics(value)
            elif kind == "error":
                raise LoopbackBenchmarkError(value, failure_diagnostics=diagnostics)
    finally:
        receive.close()
        process.join(timeout=2)
        if process.is_alive():
            forced = True
            # The worker's SIGTERM handler closes the ordinary role supervisor.
            process.terminate()
            process.join(timeout=35)
        if process.is_alive():
            import psutil
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:
                children = []
            for child in children:
                try:
                    child.kill()
                except psutil.Error:
                    pass
            process.kill()
            process.join(timeout=5)
    # Child source/supervisor close preceded its report. Parent join now covers
    # Python/native finalization too; the owner alone reaped and retained wait4.
    return finalize_client_report(report, process, forced=forced)
