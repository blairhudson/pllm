"""Paper-derived assurance controls, separate from inference components."""

from dataclasses import dataclass
from typing import ClassVar, Sequence

from pllm.components._planned import PendingComponent as PendingMethod, planned as pending


@pending("breaking-euston")
class SubspaceLeakageRegression(PendingMethod):
    pass


@pending("euston")
class EustonVariantReview(PendingMethod):
    pass


@pending("sok-transformers")
class PrivateTransformerTaxonomy(PendingMethod):
    pass


@pending("game-of-arrows")
class WeightObfuscationRegression(PendingMethod):
    pass


@dataclass(frozen=True, slots=True)
class PublicSubspaceLeakageWitness:
    """An exact public linear projection of a masked input that exposes input parity."""

    indices: tuple[int, ...]
    leaked_parity: int
    public_rank_mod2: int
    input_width: int
    mask_count: int


@dataclass(frozen=True, slots=True)
class PublicSubspaceMaskRegression:
    """Find a public mod-2 annihilator of masks in an exact 2^k ring.

    This proves leakage when a witness exists. No witness does not prove privacy:
    non-linear or higher-ideal attacks and the mask distribution remain untested.
    """

    ring_bits: int = 16
    paper_id: ClassVar[str] = "carnival"
    paper_url: ClassVar[str] = "https://pllm.run/research/papers/carnival/"
    attack_url: ClassVar[str] = "https://pllm.run/research/papers/maverick/"

    def __post_init__(self) -> None:
        if type(self.ring_bits) is not int or self.ring_bits not in (16, 24, 32):
            raise ValueError("ring_bits must be one of PLLM's exact u16, u24, or u32 rings")

    def evaluate(
        self,
        public_mask_vectors: Sequence[Sequence[int]],
        masked_input: Sequence[int],
    ) -> PublicSubspaceLeakageWitness | None:
        """Return a concrete leaked-parity witness, or None if none is found.

        The caller must establish that every actual mask is generated from the
        supplied public vectors. The masked input and basis are never retained.
        """
        if not isinstance(masked_input, (tuple, list)) or not 1 <= len(masked_input) <= 4096:
            raise ValueError("masked_input must have 1 to 4096 ring elements")
        width = len(masked_input)
        if (
            not isinstance(public_mask_vectors, (tuple, list))
            or len(public_mask_vectors) > 4096
            or width * len(public_mask_vectors) > 1_048_576
        ):
            raise ValueError("public mask basis exceeds its bounded dimensions")

        modulus = 1 << self.ring_bits
        if any(type(value) is not int or not 0 <= value < modulus for value in masked_input):
            raise ValueError("masked_input contains an invalid ring element")

        pivots: dict[int, int] = {}
        for vector in public_mask_vectors:
            if not isinstance(vector, (tuple, list)) or len(vector) != width:
                raise ValueError("public mask basis vector has the wrong width")
            row = 0
            for index, value in enumerate(vector):
                if type(value) is not int or not 0 <= value < modulus:
                    raise ValueError("public mask basis contains an invalid ring element")
                row |= (value & 1) << index
            while row:
                lead = row.bit_length() - 1
                if lead not in pivots:
                    pivots[lead] = row
                    break
                row ^= pivots[lead]

        free = next((index for index in range(width) if index not in pivots), None)
        if free is None:
            return None
        witness = 1 << free
        # Each pivot row has no set bits above its pivot, so resolve upwards.
        for lead, row in sorted(pivots.items()):
            if (row & witness).bit_count() & 1:
                witness |= 1 << lead
        indices = tuple(index for index in range(width) if witness & (1 << index))
        leaked_parity = sum(masked_input[index] for index in indices) & 1
        return PublicSubspaceLeakageWitness(
            indices=indices,
            leaked_parity=leaked_parity,
            public_rank_mod2=len(pivots),
            input_width=width,
            mask_count=len(public_mask_vectors),
        )


__all__ = [
    "SubspaceLeakageRegression", "EustonVariantReview", "PrivateTransformerTaxonomy",
    "WeightObfuscationRegression", "PublicSubspaceMaskRegression",
    "PublicSubspaceLeakageWitness",
]
