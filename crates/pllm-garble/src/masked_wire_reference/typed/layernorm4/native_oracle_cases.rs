// D43 signed-12 inputs/affine, fractional scale 12; expected outputs are i64.
// Existing Python sigma_norm only: 10 ROWS x 2 AFFINE_PAIRS, then 2 controls.
// 22 comparisons, 21 distinct triples; the first control intentionally repeats.
// Main paper/reproduction-contracts/test_sigma_layernorm4_d43.py SHA256:
// 9eb275ab30292d98ac278f6a98b936faa74b281993870e30e32cb37749b9cb57
// Main paper/reproduction-contracts/shared_profile_oracle.py SHA256:
// 2cf05b0a3df3ee99411b78b965200b152d40fbe7d628b498766ff6d44f31971c
// Expected-vector arithmetic only; no native/security/SDK qualification.

pub(super) struct Case {
    pub(super) x: [i16; 4],
    pub(super) gamma: [i16; 4],
    pub(super) beta: [i16; 4],
    pub(super) expected: [i64; 4],
}

pub(super) const CASES: [Case; 22] = [
    Case {
        x: [0, 0, 0, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [4095, -4096, 1, -1],
    },
    Case {
        x: [0, 0, 0, 0],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-1, 1, -4096, 4095],
    },
    Case {
        x: [4095, 4095, 4095, 4095],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [4095, -4096, 1, -1],
    },
    Case {
        x: [4095, 4095, 4095, 4095],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-1, 1, -4096, 4095],
    },
    Case {
        x: [-4096, -4096, -4096, -4096],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [4095, -4096, 1, -1],
    },
    Case {
        x: [-4096, -4096, -4096, -4096],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-1, 1, -4096, 4095],
    },
    Case {
        x: [-1, -1, 0, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [4095, -4096, 2, 4726],
    },
    Case {
        x: [-1, -1, 0, 0],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-1, 1, -6461, -634],
    },
    Case {
        x: [-1, 0, 0, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [4095, -4097, 2, 4094],
    },
    Case {
        x: [-1, 0, 0, 0],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-1, 2049, -6144, -1],
    },
    Case {
        x: [-2, 0, 0, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [7759, -4097, 1, 3661],
    },
    Case {
        x: [-2, 0, 0, 0],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-3665, 1832, -5928, 432],
    },
    Case {
        x: [-4, -4, 4, 3],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [7268, -4096, 2, 4227],
    },
    Case {
        x: [-4, -4, 4, 3],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-3174, -1586, -6740, -135],
    },
    Case {
        x: [-4, -4, 4, 4],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [8160, -4096, 1, 4062],
    },
    Case {
        x: [-4, -4, 4, 4],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-4066, -2032, -6128, 31],
    },
    Case {
        x: [-4096, 4095, -1, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [9897, -4098, 1, -1],
    },
    Case {
        x: [-4096, 4095, -1, 0],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-5802, 2902, -4096, 4094],
    },
    Case {
        x: [-3, 0, 1, 3],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [9591, -4096, 1, 5492],
    },
    Case {
        x: [-3, 0, 1, 3],
        gamma: [4095, 2048, -2048, -4096],
        beta: [-1, 1, -4096, 4095],
        expected: [-5496, 1, -5012, -1400],
    },
    Case {
        x: [-1, 0, 0, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [4095, -4097, 2, 4094],
    },
    Case {
        x: [-1, 1, 1, 0],
        gamma: [-4096, -1, 1, 4095],
        beta: [4095, -4096, 1, -1],
        expected: [8191, -4097, 2, -1],
    },
];
