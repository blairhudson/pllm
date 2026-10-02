"""Offline planning and native validation CLI adapters."""

from pllm.compiler import plan
from pllm.deployment import NetworkSnapshot
from pllm.plan import PlanningResult
from pllm.search import PlanningRequest

from .errors import ResolutionError
from .network import emit, load_record, write_record


def run(args, output_format, dry_run):
    command = args.plan_command
    if command == "create":
        request = load_record(PlanningRequest, args.REQUEST)
        snapshot = load_record(NetworkSnapshot, args.snapshot)
        try:
            result = plan(request, snapshot=snapshot)
        except (ValueError, TypeError) as exc:
            raise ResolutionError("PLAN_CONFIGURATION", str(exc)) from exc
        data = {
            "plan": result.to_spec(),
            **write_record(result, args.output, force=args.force, dry_run=dry_run),
        }
    else:
        result = load_record(PlanningResult, args.PLAN)
        if command == "validate":
            import time

            snapshot = load_record(NetworkSnapshot, args.snapshot)
            try:
                placement = result.validate(
                    snapshot=snapshot, evaluated_at_ms=time.time_ns() // 1_000_000
                )
            except (ValueError, TypeError) as exc:
                raise ResolutionError("PLAN_VALIDATION", str(exc)) from exc
            data = {"valid": True, "result_digest": result.digest, "placement": placement}
        elif command == "explain":
            document = result.to_spec()
            data = {
                key: document[key]
                for key in (
                    "status",
                    "coverage",
                    "selection",
                    "rejections",
                    "alternatives",
                    "privacy_scope",
                    "policy_digest",
                )
            }
            data["policy"] = result.request.policy.to_spec()
        else:
            data = {"plan": result.to_spec()}
    emit(f"plan.{command}", {**data, "dry_run": dry_run}, args, output_format)
