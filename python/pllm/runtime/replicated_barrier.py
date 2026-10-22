"""Drain existing three-role frozen waves without changing their wire schedule.

The caller's all_roles operation must await every role, including delivery ACKs.
This establishes router drain only, not native idleness or an admitted body entry.
"""


async def drain_peer_waves(all_roles):
    for _ in range(1024):
        # Freeze every sender before sending. Replies from a fast peer belong to
        # the next wave and cannot overtake a slow peer's prerequisite opening.
        sent = await all_roles("prepare_dispatch")
        if any(set(state) != {"sent"} or type(state["sent"]) is not int
               or not 0 <= state["sent"] <= 16 for state in sent):
            raise ValueError("replicated dispatch acknowledgement differs")
        if not any(state["sent"] for state in sent):
            return
        delivered = await all_roles("dispatch")
        if delivered != sent:
            raise ValueError("replicated peer wave count differs")
    raise ValueError("replicated peer round bound exceeded")
