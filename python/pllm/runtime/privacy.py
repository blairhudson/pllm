from __future__ import annotations

from enum import Enum


class PrivacyMode(str, Enum):
    """The privacy setting selected when the server starts."""

    PUBLIC = "public"
    PROPRIETARY = "proprietary"

    @classmethod
    def parse(cls, value: str | "PrivacyMode") -> "PrivacyMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"privacy mode must be one of: {supported}") from exc

    @property
    def protocol(self) -> str:
        return "masked_w4a4" if self is self.PUBLIC else "guarded_w4a4"

    @property
    def description(self) -> str:
        if self is self.PUBLIC:
            return (
                "Fast private inference for public weights. HE prepares fresh single use "
                "correlations before the online request."
            )
        return (
            "Private inference for proprietary weights. Guarded mode limits layer queries. "
            "Secure mode is a research preview that keeps intermediate values split."
        )


class ProprietaryProtocol(str, Enum):
    """The protocol used behind proprietary mode."""

    GUARDED = "guarded"
    BLINDED = "blinded"
    SECURE = "secure"
    DIRECT = "direct"

    @classmethod
    def parse(cls, value: str | "ProprietaryProtocol") -> "ProprietaryProtocol":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("_", "-")
        aliases = {
            "fast": cls.GUARDED,
            "hardened": cls.GUARDED,
            "guarded-blinded": cls.GUARDED,
            "blinded-ole": cls.BLINDED,
            "mpc": cls.SECURE,
            "malicious": cls.SECURE,
            "direct": cls.DIRECT,
            "fhe": cls.DIRECT,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"proprietary protocol must be one of: {supported}") from exc
