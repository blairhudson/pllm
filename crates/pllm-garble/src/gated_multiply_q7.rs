//! Substitutable implementations of protected Q7 `SiLU(gate) * up`.

pub const BINARY_TABLE_COMPONENT_ID: &str = "pllm/binary-table/v1";
pub const R03_CRT_COMPONENT_ID: &str = "pllm/r03-crt/v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    BinaryTable,
    R03Crt,
}

impl Method {
    pub const fn component_id(self) -> &'static str {
        match self {
            Self::BinaryTable => BINARY_TABLE_COMPONENT_ID,
            Self::R03Crt => R03_CRT_COMPONENT_ID,
        }
    }
}
