"""Paper-derived assurance controls and threat-model checks.

These declared controls are not inference components; they raise until an
independently tested assurance implementation is available.
"""

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


@pending("carnival")
class PublicSubspaceMaskRegression(PendingMethod):
    pass


__all__ = [
    "SubspaceLeakageRegression", "EustonVariantReview", "PrivateTransformerTaxonomy",
    "WeightObfuscationRegression", "PublicSubspaceMaskRegression",
]
