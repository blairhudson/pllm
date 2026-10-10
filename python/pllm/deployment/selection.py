"""Shared lazy selected-execution owner for SDK, gateway and benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import threading
import time
from pathlib import Path

from pllm.compiler import plan, plan_on_load
from pllm.configuration import Experiment
from pllm.deployment.execution import open_execution
from pllm.deployment.network import NetworkError, NetworkSpec, discover, strict_load
from pllm.search.placement import PlanningPolicy


class SelectedClientFactory:
    """Fresh capacity per attempt; cached immutable planning inputs only.

    An explicit Experiment pins its Pipeline. Source resolution is lazy and the
    native planner remains the authority for every selected role assignment.
    """

    def __init__(self, network, *, result=None, request=None, experiment=None,
                 policy=None, credentials=None, ttl_seconds=60):
        if not isinstance(network, NetworkSpec):
            raise TypeError("network must be NetworkSpec")
        if sum(value is not None for value in (result, request, experiment)) != 1:
            raise ValueError("choose exactly one plan, planning request or Experiment")
        if policy is not None and (experiment is None or not isinstance(policy, PlanningPolicy)):
            raise TypeError("policy requires an Experiment and PlanningPolicy")
        if experiment is not None and not isinstance(experiment, Experiment):
            raise TypeError("experiment must be Experiment")
        if type(ttl_seconds) is not int or not 5 <= ttl_seconds <= 300:
            raise ValueError("lease lifetime must be in [5,300] seconds")
        self.network, self.result, self.request = network, result, request
        self.experiment, self.policy = experiment, policy or PlanningPolicy()
        self.credentials, self.ttl_seconds = credentials, ttl_seconds
        self._lock = threading.Lock()
        candidates = ((experiment,) if experiment is not None else
                      (result.experiment,) if result is not None else request.candidates)
        if any(item is None for item in candidates):
            raise NetworkError("NO_FEASIBLE_PLACEMENT")
        for candidate in candidates:
            intent = candidate.deployment
            if network.backend == "http":
                if (intent.kind != "network" or intent.network_id != network.network_id
                        or intent.network_spec_digest != network.digest):
                    raise NetworkError("selected candidate network intent differs")
            elif intent.kind != "local":
                raise NetworkError("local backend requires local deployment")
        self.model_ids = tuple(sorted({item.resolve().model for item in candidates}))
        self.default_model = self.model_ids[0] if len(self.model_ids) == 1 else None

    @classmethod
    def from_settings(cls, settings):
        if not settings.network or not settings.experiment:
            raise NetworkError("automatic planning requires client network and experiment paths")
        network = NetworkSpec.from_spec(strict_load(Path(settings.network).read_bytes()))
        experiment = Experiment.from_spec(strict_load(Path(settings.experiment).read_bytes()))
        policy = PlanningPolicy(**(settings.planning or {}))
        factory = cls(network, experiment=experiment, policy=policy)
        if settings.model is not None and settings.model not in factory.model_ids:
            raise NetworkError("client model conflicts with configured deployment")
        return factory

    def select(self):
        if self.result is not None:
            return self.result
        snapshot = discover(self.network, credentials=self.credentials)
        stamp = time.time_ns() // 1_000_000
        with self._lock:
            if self.request is None:
                from pllm.search import SearchSpace

                policy = replace(self.policy, evaluated_at_ms=stamp)
                result = plan_on_load(self.experiment, snapshot=snapshot, policy=policy,
                                      space=SearchSpace(self.experiment, {}))
                self.request = result.request
            else:
                request = replace(self.request, policy=replace(
                    self.request.policy, evaluated_at_ms=stamp))
                result = plan(request, snapshot=snapshot)
        if result.status != "feasible":
            error = NetworkError("SEARCH_LIMIT" if result.status == "inconclusive"
                                 else "NO_FEASIBLE_PLACEMENT")
            error.planning_result = result
            raise error
        return result

    @contextmanager
    def open(self, *, cancelled=None):
        from pllm.runtime.client import OpenAI

        result = self.select()
        if cancelled is not None and cancelled.is_set():
            raise NetworkError("request cancelled before capacity reservation")
        with open_execution(result, network=self.network, credentials=self.credentials,
                            ttl_seconds=self.ttl_seconds) as lease:
            with OpenAI(execution=lease) as client:
                yield client, result

    @contextmanager
    def __call__(self):
        with self.open() as (client, _result):
            yield client
