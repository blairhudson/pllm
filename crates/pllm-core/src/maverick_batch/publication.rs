use sha2::{Digest, Sha256};

use super::arithmetic::canonical;
use super::code::{Code, CodeDescription};
use super::memory::{AllocationPlan, Memory, MemoryStats, Words};
use super::{
    Error, Result, BATCH_SIZE, DIMENSION, FIELD, LENGTH, SECONDARY_DIMENSION, SECONDARY_LENGTH,
};

#[derive(Clone, Copy)]
pub struct SetupInputs<'a> {
    pub matrices: [&'a [u32]; BATCH_SIZE],
    pub privacy_codes: [CodeDescription<'a>; BATCH_SIZE],
    pub primary_code: CodeDescription<'a>,
    pub secondary_code: CodeDescription<'a>,
}

impl SetupInputs<'_> {
    fn validate(self) -> Result<()> {
        for matrix in self.matrices {
            canonical(matrix, DIMENSION * DIMENSION)?;
        }
        for code in self.privacy_codes {
            code.validate(DIMENSION)?;
        }
        self.primary_code.validate(DIMENSION)?;
        self.secondary_code.validate(SECONDARY_DIMENSION)
    }
}

/// Full incoming public state; borrowed lengths are checked before copying.
#[derive(Clone, Copy)]
pub struct PublicationParts<'a> {
    pub p: [&'a [u32]; BATCH_SIZE],
    pub q: [&'a [u32]; BATCH_SIZE],
    pub q_auxiliary: &'a [u32],
}

impl PublicationParts<'_> {
    fn validate_shape(self) -> Result<()> {
        for p in self.p {
            canonical(p, DIMENSION * LENGTH)?;
        }
        for q in self.q {
            canonical(q, LENGTH * DIMENSION)?;
        }
        canonical(self.q_auxiliary, LENGTH * SECONDARY_LENGTH)
    }
}

#[derive(Clone, Copy)]
pub struct PublicationView<'a> {
    pub source: SetupInputs<'a>,
    pub derived: PublicationParts<'a>,
    pub binding: [u8; 32],
}

pub(super) struct Material {
    pub(super) memory: Memory,
    pub(super) matrices: [Words; BATCH_SIZE],
    pub(super) gx: [Code; BATCH_SIZE],
    pub(super) gy: Code,
    pub(super) gp: Code,
    pub(super) p: [Words; BATCH_SIZE],
    pub(super) q: [Words; BATCH_SIZE],
    pub(super) aux: Words,
    pub(super) binding: [u8; 32],
}

impl Material {
    pub(super) fn view(&self) -> PublicationView<'_> {
        PublicationView {
            source: SetupInputs {
                matrices: [self.matrices[0].as_slice(), self.matrices[1].as_slice()],
                privacy_codes: [self.gx[0].description(), self.gx[1].description()],
                primary_code: self.gy.description(),
                secondary_code: self.gp.description(),
            },
            derived: PublicationParts {
                p: [self.p[0].as_slice(), self.p[1].as_slice()],
                q: [self.q[0].as_slice(), self.q[1].as_slice()],
                q_auxiliary: self.aux.as_slice(),
            },
            binding: self.binding,
        }
    }

    fn bind(&mut self) -> Result<()> {
        let mut hash = Sha256::new();
        hash.update(b"pllm.maverick.babybear.explicit-raa.batch2.algebra.v1\0");
        hash.update(FIELD.to_le_bytes());
        for value in [64_u64, 64, 2, 512, 1024, 165, 31, 2, 31, 2] {
            hash.update(value.to_le_bytes());
        }
        for code in [&self.gx[0], &self.gx[1], &self.gy, &self.gp] {
            hash.update(code.description().digest()?);
        }
        for matrix in &self.matrices {
            let mut matrix_hash = Sha256::new();
            matrix_hash.update(b"pllm.maverick.matrix.v1\0");
            matrix_hash.update(FIELD.to_le_bytes());
            matrix_hash.update((DIMENSION as u64).to_le_bytes());
            matrix_hash.update((DIMENSION as u64).to_le_bytes());
            for &value in matrix.as_slice() {
                matrix_hash.update(value.to_le_bytes());
            }
            hash.update(matrix_hash.finalize());
        }
        for public in [&self.p[0], &self.p[1], &self.q[0], &self.q[1], &self.aux] {
            for &value in public.as_slice() {
                hash.update(value.to_le_bytes());
            }
        }
        self.binding = hash.finalize().into();
        Ok(())
    }
}

/// Cannot create client/server owners before complete validation succeeds.
pub struct UnvalidatedPublication {
    material: Material,
}

pub struct ValidatedPublication {
    pub(super) material: Material,
}

impl UnvalidatedPublication {
    pub fn import(
        plan: AllocationPlan,
        source: SetupInputs<'_>,
        parts: PublicationParts<'_>,
    ) -> Result<Self> {
        source.validate()?;
        parts.validate_shape()?;
        let memory = plan.start();
        let mut material = Material {
            matrices: [
                Words::copy(&memory, source.matrices[0])?,
                Words::copy(&memory, source.matrices[1])?,
            ],
            gx: [
                Code::copy(&memory, source.privacy_codes[0])?,
                Code::copy(&memory, source.privacy_codes[1])?,
            ],
            gy: Code::copy(&memory, source.primary_code)?,
            gp: Code::copy(&memory, source.secondary_code)?,
            p: [
                Words::copy(&memory, parts.p[0])?,
                Words::copy(&memory, parts.p[1])?,
            ],
            q: [
                Words::copy(&memory, parts.q[0])?,
                Words::copy(&memory, parts.q[1])?,
            ],
            aux: Words::copy(&memory, parts.q_auxiliary)?,
            binding: [0; 32],
            memory,
        };
        material.bind()?;
        Ok(Self { material })
    }

    pub fn view(&self) -> PublicationView<'_> {
        self.material.view()
    }

    pub fn memory_stats(&self) -> MemoryStats {
        self.material.memory.stats()
    }

    /// Validate against independently selected intended matrices AND codes.
    /// Every P, Q and Qaux entry is recomputed; no sampled spot-check admission.
    pub fn validate(self, intended: SetupInputs<'_>) -> Result<ValidatedPublication> {
        intended.validate()?;
        let m = &self.material;
        for b in 0..BATCH_SIZE {
            if intended.matrices[b] != m.matrices[b].as_slice()
                || !intended.privacy_codes[b].same_as(m.gx[b].description())
            {
                return Err(Error::Binding);
            }
        }
        if !intended.primary_code.same_as(m.gy.description())
            || !intended.secondary_code.same_as(m.gp.description())
        {
            return Err(Error::Binding);
        }
        for b in 0..BATCH_SIZE {
            for row in 0..DIMENSION {
                let encoded = m.gx[b].encode(
                    &m.memory,
                    &intended.matrices[b][row * DIMENSION..][..DIMENSION],
                )?;
                if encoded.as_slice() != &m.p[b].as_slice()[row * LENGTH..][..LENGTH] {
                    return Err(Error::Publication);
                }
            }
            let mut column = Words::zeros(&m.memory, DIMENSION)?;
            for col in 0..DIMENSION {
                for row in 0..DIMENSION {
                    column.as_mut()[row] = intended.matrices[b][row * DIMENSION + col];
                }
                let encoded = m.gy.encode(&m.memory, column.as_slice())?;
                for row in 0..LENGTH {
                    if encoded.as_slice()[row] != m.q[b].as_slice()[row * DIMENSION + col] {
                        return Err(Error::Publication);
                    }
                }
            }
        }
        let mut stacked_row = Words::zeros(&m.memory, SECONDARY_DIMENSION)?;
        for row in 0..LENGTH {
            for b in 0..BATCH_SIZE {
                stacked_row.as_mut()[b * DIMENSION..][..DIMENSION]
                    .copy_from_slice(&m.q[b].as_slice()[row * DIMENSION..][..DIMENSION]);
            }
            let encoded = m.gp.encode(&m.memory, stacked_row.as_slice())?;
            if encoded.as_slice() != &m.aux.as_slice()[row * SECONDARY_LENGTH..][..SECONDARY_LENGTH]
            {
                return Err(Error::Publication);
            }
        }
        Ok(ValidatedPublication {
            material: self.material,
        })
    }
}

impl ValidatedPublication {
    pub fn view(&self) -> PublicationView<'_> {
        self.material.view()
    }

    pub fn memory_stats(&self) -> MemoryStats {
        self.material.memory.stats()
    }
}

/// Independent source-specific preprocessing; does not execute any upstream code.
pub fn preprocess(plan: AllocationPlan, source: SetupInputs<'_>) -> Result<UnvalidatedPublication> {
    source.validate()?;
    let memory = plan.start();
    let matrices = [
        Words::copy(&memory, source.matrices[0])?,
        Words::copy(&memory, source.matrices[1])?,
    ];
    let gx = [
        Code::copy(&memory, source.privacy_codes[0])?,
        Code::copy(&memory, source.privacy_codes[1])?,
    ];
    let gy = Code::copy(&memory, source.primary_code)?;
    let gp = Code::copy(&memory, source.secondary_code)?;
    let mut p = [
        Words::zeros(&memory, DIMENSION * LENGTH)?,
        Words::zeros(&memory, DIMENSION * LENGTH)?,
    ];
    let mut q = [
        Words::zeros(&memory, LENGTH * DIMENSION)?,
        Words::zeros(&memory, LENGTH * DIMENSION)?,
    ];
    for b in 0..BATCH_SIZE {
        for row in 0..DIMENSION {
            let encoded = gx[b].encode(
                &memory,
                &matrices[b].as_slice()[row * DIMENSION..][..DIMENSION],
            )?;
            p[b].as_mut()[row * LENGTH..][..LENGTH].copy_from_slice(encoded.as_slice());
        }
        let mut column = Words::zeros(&memory, DIMENSION)?;
        for col in 0..DIMENSION {
            for row in 0..DIMENSION {
                column.as_mut()[row] = matrices[b].as_slice()[row * DIMENSION + col];
            }
            let encoded = gy.encode(&memory, column.as_slice())?;
            for row in 0..LENGTH {
                q[b].as_mut()[row * DIMENSION + col] = encoded.as_slice()[row];
            }
        }
    }
    let mut aux = Words::zeros(&memory, LENGTH * SECONDARY_LENGTH)?;
    let mut stacked_row = Words::zeros(&memory, SECONDARY_DIMENSION)?;
    for row in 0..LENGTH {
        for b in 0..BATCH_SIZE {
            stacked_row.as_mut()[b * DIMENSION..][..DIMENSION]
                .copy_from_slice(&q[b].as_slice()[row * DIMENSION..][..DIMENSION]);
        }
        let encoded = gp.encode(&memory, stacked_row.as_slice())?;
        aux.as_mut()[row * SECONDARY_LENGTH..][..SECONDARY_LENGTH]
            .copy_from_slice(encoded.as_slice());
    }
    let mut material = Material {
        memory,
        matrices,
        gx,
        gy,
        gp,
        p,
        q,
        aux,
        binding: [0; 32],
    };
    material.bind()?;
    Ok(UnvalidatedPublication { material })
}
