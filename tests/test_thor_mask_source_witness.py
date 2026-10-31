"""Manual source-level mathematical witness, not an upstream execution test.

Source: crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25,
src/thor/data.py:69–88, ThorDataEncryptor.encode_attention_mask. Source and
notebook caller attachment are documented in
docs/research/thor-multiblock-indexing-audit.md (Astra29).

Witness bodies use only the standard library and model pre-encoding message
support. Ordinary pytest defaults load repository fixtures and runtime imports.
For an isolated invocation from the repository root, use:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest -p no:cacheprovider tests/test_thor_mask_source_witness.py
No upstream imports, vendoring, source-file/hash gates, HE, numeric, or Torch
execution occur in the witness bodies; no encoded-object equality, effective
dtype, or plaintext-oracle equivalence is claimed. This is not a general mask
encoder API.

Additional witnesses cover source-fixed-extent-128 geometry only. The logical
oracle scatters allowed key/query pairs through the documented diagonal layout
(audit lines 24–28, 49–52), independently of the pinned class threshold scan.
No decoder packing approval or nonempty inverse-domain validity is implied.
"""


def _source_message_support(mask):
    # Count nonzero entries, not their positions. Addresses retain 16 head
    # slots per token, of which only 12 are active in the pinned source.
    k = sum(value != 0 for value in mask)
    return (
        (i, 2048 * j + 16 * t + h)
        for i in range(8)
        for j in range(16)
        for t in range(128)
        for h in range(12)
        if (16 * i + j + t) % 128 < k
    )


def test_same_count_distinct_token_supports_collide_before_encoding():
    first = tuple(int(position == 0) for position in range(128))
    last = tuple(int(position == 127) for position in range(128))
    first_intended = {position for position, value in enumerate(first) if value}
    last_intended = {position for position, value in enumerate(last) if value}

    assert first_intended == {0}
    assert last_intended == {127}
    assert first_intended != last_intended
    assert sum(value != 0 for value in first) == 1
    assert sum(value != 0 for value in last) == 1

    # Independent k=1 modular solution, not another count/threshold scan.
    # These are message addresses, not original token-position indices.
    expected = {
        (i, 2048 * j + 16 * ((-16 * i - j) % 128) + h)
        for i in range(8)
        for j in range(16)
        for h in range(12)
    }
    first_support = set(_source_message_support(first))
    last_support = set(_source_message_support(last))
    assert first_support == last_support == expected
    assert len(expected) == 8 * 16 * 12
    assert all(slot % 16 < 12 for _, slot in first_support)


def test_empty_count_has_no_message_support():
    assert set(_source_message_support((0,) * 128)) == set()


def test_full_count_occupies_only_twelve_of_sixteen_head_slots():
    # Stream full occupancy instead of retaining a large support set.
    occupancy = 0
    head_counts = [0] * 16
    for frame, slot in _source_message_support((1,) * 128):
        assert 0 <= frame < 8
        assert 0 <= slot < 32768
        occupancy += 1
        head_counts[slot % 16] += 1

    assert occupancy == 8 * 16 * 128 * 12
    assert head_counts == [8 * 16 * 128] * 12 + [0] * 4


def _logical_key_query_addresses(key, query):
    # Definition 3.1: L_d(A)[query] = A[(d + query) % 128, query].
    # Here the first coordinate is the allowed key, the second the query.
    # Solve for the diagonal, then interlace 16 physical head lanes and
    # concatenate 16 diagonals per frame (audit lines 24–28, 49–52).
    diagonal = (key - query) % 128
    frame, local_diagonal = divmod(diagonal, 16)
    start = (local_diagonal * 128 + query) * 16
    return frame, start


def _logical_dense_prefix_messages(k):
    messages = [bytearray(32768) for _ in range(8)]
    # Truth starts with logical allowed keys, not a physical-slot threshold.
    for key in range(k):
        for query in range(128):
            frame, start = _logical_key_query_addresses(key, query)
            for head in range(12):
                messages[frame][start + head] = 1
    return messages


def test_source_fixed_extent_128_dense_prefix_geometry():
    for k in (1, 17, 127, 128):
        logical = _logical_dense_prefix_messages(k)
        literal = [bytearray(32768) for _ in range(8)]
        mask = tuple(int(key < k) for key in range(128))
        # Preserve the pinned class's actual formula as the source oracle.
        for frame, slot in _source_message_support(mask):
            literal[frame][slot] = 1

        for frame in range(8):
            # Full message equality checks exact zero everywhere else, not
            # encoded HE-object equality or merely the number of set lanes.
            assert literal[frame] == logical[frame], (k, frame)
            assert sum(literal[frame]) == 16 * k * 12
            assert [sum(literal[frame][head::16]) for head in range(16)] == (
                [16 * k] * 12 + [0] * 4
            )


def test_source_fixed_extent_128_two_query_causal_prefill_requires_mismatch():
    queries = (0, 16)
    boundary_cases = {
        0: ([False, False], [set(), set()], [True, True]),
        1: ([True, False], [{0}, set()], [True, False]),
        16: ([True, False], [{0}, set()], [False, False]),
        17: ([True, True], [{0}, set(range(17))], [False, True]),
        128: ([True, True], [{0}, set(range(17))], [False, False]),
    }

    # Exhaust supported dense-prefix lengths. Logical causal eligibility is
    # active(query) AND key < k AND key <= query, with active(query)=query<k.
    # Only k>=17 makes BOTH selected queries active; inactive-query population
    # is a separate discrepancy, not evidence of incompatible active rows.
    for k in range(129):
        active = [query < k for query in queries]
        causal_allowed = [
            {key for key in range(128) if query < k and key < k and key <= query}
            for query in queries
        ]
        mask = tuple(int(key < k) for key in range(128))
        # Keep only the two query columns while streaming the literal map.
        source_support = {
            (frame, slot)
            for frame, slot in _source_message_support(mask)
            if (slot // 16) % 128 in queries
        }
        row_matches = []
        source_rows = []
        for query, allowed in zip(queries, causal_allowed):
            actual = set()
            expected = set()
            source_keys = set()
            for key in range(128):
                frame, start = _logical_key_query_addresses(key, query)
                for head in range(16):
                    address = (frame, start + head)
                    if address in source_support:
                        actual.add(address)
                        if head == 0:
                            source_keys.add(key)
                    if key in allowed and head < 12:
                        expected.add(address)
            row_matches.append(actual == expected)
            source_rows.append(source_keys)

        assert source_rows == [set(range(k)), set(range(k))], k
        assert row_matches == [k in (0, 1), k in (0, 17)], k
        if k in boundary_cases:
            assert (active, causal_allowed, row_matches) == boundary_cases[k], k

        if k == 0:
            assert not source_support
            assert causal_allowed == [set(), set()]
            assert all(row_matches)
        elif k <= 16:
            assert active == [True, False]
            assert causal_allowed == [{0}, set()]
            # Source populates inactive query 16. For k=1, the only active
            # query matches: this is not two-active-row causal incompatibility.
            assert source_rows[1]
            assert not row_matches[1]
        else:
            assert all(active)
            assert causal_allowed == [{0}, set(range(17))]
            assert causal_allowed[0] != causal_allowed[1]
            # Both active rows require distinct key prefixes, whereas one
            # fixed source vector supplies the same prefix to both rows.
            assert source_rows[0] == source_rows[1]
            assert not all(row_matches), k
