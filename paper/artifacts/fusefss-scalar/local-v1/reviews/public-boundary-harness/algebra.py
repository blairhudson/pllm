"""Finite independent ideal-algebra checks, NOT cryptographic backend tests."""
import json


def d(bits, r, u, p):
    """Ideal unit DCF equation on gamma=r-1, not native DCF evaluation."""
    n = 1 << bits
    return int((u - 1 - p) % n < (r - 1) % n) - int(u > p % n)


def correction(bits, r, p):
    n = 1 << bits
    return int((r + p % n) % n > p % n) - int(r > 0) + int(r + p >= n)


def check_comparison(bits, r, u, p):
    n = 1 << bits
    actual = (d(bits, r, u, p) - d(bits, r, u, 0) + correction(bits, r, p)) % n
    assert actual == int((u - r) % n < p), (bits, r, u, p, actual)


def main():
    comparisons = 0
    for bits in range(1, 6):
        n = 1 << bits
        for r in range(n):
            for u in range(n):
                for p in range(n + 1):
                    check_comparison(bits, r, u, p)
                    comparisons += 1
    endpoint_comparisons = 0
    n = 1 << 50
    for r in [0, 1, 63, 64, n // 2, n - 64, n - 1]:
        for u in [0, 1, 63, 64, n // 2, n - 64, n - 1]:
            for p in [0, 1, 63, 64, n // 2, n - 64, n - 1, n]:
                check_comparison(50, r, u, p)
                endpoint_comparisons += 1

    partitions = cases = 0
    for bits in range(1, 4):
        n = 1 << bits
        for subset in range(1 << (n - 1)):
            source = [0] + [j for j in range(1, n) if subset & (1 << (j - 1))] + [n]
            coeff = [((7 * j + 3) % n, (3 * j * j + 5) % n) for j in range(len(source) - 1)]
            boundaries, payloads = source[:], coeff[:]
            if len(coeff) < n:
                index = next(j for j in range(len(coeff)) if source[j + 1] - source[j] >= 2)
                boundaries.insert(index + 1, source[index] + 1)
                payloads.insert(index + 1, coeff[index])
            assert len(payloads) == min(len(coeff) + 1, n)
            for r in range(n):
                corrected = list(payloads[-1])
                for p, left, right in zip(boundaries[1:-1], payloads, payloads[1:]):
                    for j in range(2):
                        corrected[j] += (left[j] - right[j]) * correction(bits, r, p)
                for x in range(n):
                    u = (x + r) % n
                    output = corrected[:]
                    for p, left, right in zip(boundaries[1:-1], payloads, payloads[1:]):
                        difference = d(bits, r, u, p) - d(bits, r, u, 0)
                        for j in range(2):
                            output[j] += (left[j] - right[j]) * difference
                    expected = next(coeff[i] for i in range(len(coeff)) if source[i] <= x < source[i + 1])
                    assert tuple(v % n for v in output) == expected
                    cases += 1
            partitions += 1
    print(json.dumps({"scope": "ideal-equations-only-not-native-FSS-or-security", "status": "pass",
                      "small_comparison_mask_input_threshold_cases": comparisons,
                      "ring50_endpoint_comparison_cases": endpoint_comparisons,
                      "all_small_complete_partitions": partitions,
                      "partition_mask_input_cases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
