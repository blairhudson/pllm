"""Two-request lifecycle budget for the ordered prepared-frontier cohort."""
from benchmarks.research import prepared_frontier


def _sequence(experiment):
    return experiment.with_params(budget=experiment.budget.with_params(requests=2))


def serial():
    return _sequence(prepared_frontier.serial())


def windowed():
    return _sequence(prepared_frontier.windowed())


def reused():
    return _sequence(prepared_frontier.reused())


def resident():
    return _sequence(prepared_frontier.resident())
