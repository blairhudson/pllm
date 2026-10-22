"""Frozen router-wave order; not native body admission or checkpoint execution."""

import asyncio

import pytest

from pllm.runtime.replicated_barrier import drain_peer_waves


def states(*counts):
    return [{"sent": count} for count in counts]


@pytest.mark.asyncio
async def test_delayed_ack_and_generated_reply_are_drained_before_return():
    pending = [[b"opening"], [], []]
    frozen = None
    actions = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def all_roles(action):
        nonlocal frozen
        actions.append(action)
        if action == "prepare_dispatch":
            frozen = [tuple(queue) for queue in pending]
            for queue in pending:
                queue.clear()
            return states(*(len(queue) for queue in frozen))
        assert action == "dispatch"
        assert frozen is not None
        if frozen[0] == (b"opening",):
            # One receiver produces a reply while another ACK remains in flight.
            # It must not join the current frozen wave or imply global drain.
            pending[2].append(b"reply")
            entered.set()
            await release.wait()
        return states(*(len(queue) for queue in frozen))

    draining = asyncio.create_task(drain_peer_waves(all_roles))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert actions == ["prepare_dispatch", "dispatch"]
        assert not draining.done()
        release.set()
        await asyncio.wait_for(draining, 1)
        assert actions == ["prepare_dispatch", "dispatch", "prepare_dispatch",
                           "dispatch", "prepare_dispatch"]
        assert pending == [[], [], []]
    finally:
        release.set()
        draining.cancel()
        await asyncio.gather(draining, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_wave_sends_no_dispatch():
    actions = []

    async def all_roles(action):
        actions.append(action)
        return states(0, 0, 0)

    await drain_peer_waves(all_roles)
    assert actions == ["prepare_dispatch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    {"sent": True}, {"sent": -1}, {"sent": 17}, {"sent": 1.0},
    {"sent": 0, "extra": 0}, {},
])
async def test_malformed_prepare_ack_never_dispatches(invalid):
    async def all_roles(action):
        assert action == "prepare_dispatch"
        return [*states(0, 0), invalid]

    with pytest.raises(ValueError, match="dispatch acknowledgement differs"):
        await drain_peer_waves(all_roles)


@pytest.mark.asyncio
async def test_changed_delivery_count_cannot_establish_drain():
    actions = []

    async def all_roles(action):
        actions.append(action)
        return states(1, 0, 0) if action == "prepare_dispatch" else states(0, 0, 0)

    with pytest.raises(ValueError, match="peer wave count differs"):
        await drain_peer_waves(all_roles)
    assert actions == ["prepare_dispatch", "dispatch"]


@pytest.mark.asyncio
async def test_existing_wave_limit_preserves_exact_bound():
    actions = []

    async def all_roles(action):
        actions.append(action)
        return states(1, 0, 0)

    with pytest.raises(ValueError, match="peer round bound exceeded"):
        await drain_peer_waves(all_roles)
    assert actions == ["prepare_dispatch", "dispatch"] * 1024


@pytest.mark.asyncio
async def test_cancelled_dispatch_does_not_announce_drain_or_start_next_wave():
    actions = []

    async def all_roles(action):
        actions.append(action)
        if action == "dispatch":
            raise asyncio.CancelledError
        return states(1, 0, 0)

    with pytest.raises(asyncio.CancelledError):
        await drain_peer_waves(all_roles)
    assert actions == ["prepare_dispatch", "dispatch"]
