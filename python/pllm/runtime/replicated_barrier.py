"""Drain existing three-role frozen waves without changing their wire schedule.

The caller's all_roles operation must await every role, including delivery ACKs.
This establishes router drain only, not native idleness or an admitted body entry.
"""


def _wave_counts(states):
    if (type(states) not in (list, tuple) or len(states) != 3
            or any(type(state) is not dict or set(state) != {"sent"}
                   or type(state["sent"]) is not int or not 0 <= state["sent"] <= 16
                   for state in states)):
        raise ValueError("replicated dispatch acknowledgement differs")
    return tuple(state["sent"] for state in states)


async def drain_peer_waves(all_roles):
    for _ in range(1024):
        # Freeze every sender before sending. Replies from a fast peer belong to
        # the next wave and cannot overtake a slow peer's prerequisite opening.
        sent = _wave_counts(await all_roles("prepare_dispatch"))
        if not any(sent):
            return
        delivered = _wave_counts(await all_roles("dispatch"))
        if delivered != sent:
            raise ValueError("replicated peer wave count differs")
    raise ValueError("replicated peer round bound exceeded")
