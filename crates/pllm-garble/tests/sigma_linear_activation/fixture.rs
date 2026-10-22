//! Immutable checkpoint-derived SUBMATRIX, not an original Qwen layer.
//! Raw source bytes only: no inference, download, or production encoder.

pub const REPOSITORY: &str = "Qwen/Qwen2.5-0.5B-Instruct";
pub const REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
// Range-only consumers do not use this pin or claim full-file verification.
#[allow(dead_code)]
pub const FILE_SHA256: &str = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe";
pub const TENSOR: &str = "model.layers.0.mlp.gate_proj.weight";
pub const SOURCE_SHAPE_OI: [usize; 2] = [4864, 896];
pub const INNER: usize = 256;
pub const SELECTED_SHA256: &str =
    "780a723ba14459a03b2e72508debd6d4551d6ca13da75534c0f456de84c680de";

// BF16 little-endian: original OI row 0, columns [0,256). Source tensor
// data_offsets=[280987392,289703680], safetensors header length=32280.
// Absolute file byte interval [281019680,281020192). FILE_SHA256 is the
// existing HF content-addressed blob pin, not a fresh full-file verification.
const BF16_HEX: &str = concat!(
    "ecbcae3cb4bbc23c2d3cee3ca03c8ebb6abc18bb283c893cb33c0dbc6b3cfd3c793b463d47bc93bc263de53b513ca83c89bb74bc043c89ba02bdf03b9abc76bc8c3c783ba9bba23d253dc4bcda3ccf3c80bb47bcd13c863c67bc4b3d37bbcabc22ba753c1dbd913c23bb69bce83c4bbcb13cd5bb2fbd89bcea3a093d9fbc55bcac3c243ca1bc653c033da13c8e3c323c74bc743c393c4b3c8cbdc038ea3719bd803d14bde8bb3ebc7b3c213b453bacbc8ebc8abc9c3b693c8f3a9eba18bc58bc0b3ca83b5fbc833c1abd523b88bb5c3d823cc6bad13cd5bcafbb80bcd33c7fbc83bcc3bcfd3ce2bcc3bc05bd113d8e3c4cbc6cbbb7bb4dbc97bc443c7f3d10bd60bb1b3c34bced3bd93cd63c9b3b383c1bbcd9bcd83c03bce73a873c433bbdb8d0bbfaba313d3c3cfc3be639abbbc63cc0bca3bb823c463cae3cabbc83bc183ca3bced3c803cfcbb1dbb283c153db33bc5bc81bb853d8d3c26bd0bbd1d3c6b3c743c093d893c043b813cc53be13b303a983c1b3cb0bb393aa73ba0bca0bc6dbc62bd813cc33c073de6bb61bccfb9cb3ca6b9dcb8273b053d31bb69bc4dbd873c64bceab954bbe03b2bbb9a3c52bc653b273c8c3c703c253c5d3c84bc713c9ebc243db7bc703c493ca03c833bbdbbabbc063ddb3cf7bb2ebb473d2d3c1a3c1538cfbb0c3df33b683c693cd83c1c3d71bc86bc523c06bd08bdbb39e83c043d1bbc",
);

pub fn source_bytes() -> Vec<u8> {
    (0..BF16_HEX.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&BF16_HEX[i..i + 2], 16).unwrap())
        .collect()
}

/// Exact widening of BF16 to binary32, then binary64. OI [1,256] becomes
/// right-hand IO [256,1]; same contiguous order because output width is one.
pub fn weights_io() -> Vec<f64> {
    source_bytes()
        .chunks_exact(2)
        .map(|b| {
            f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
            ))
        })
        .collect()
}

/// Public synthetic test inputs, NOT checkpoint hidden activations or an
/// empirical activation range. Declared domain is real [-1,1], inclusive.
pub fn input() -> Vec<f64> {
    (0..INNER)
        .map(|i| ((i * 73 % 8193) as f64 - 4096.0) / 4096.0)
        .collect()
}
