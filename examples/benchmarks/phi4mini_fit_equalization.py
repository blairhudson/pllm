"""Build the source-locked profile from a small fixed public token cohort."""

from __future__ import annotations

import os
from pathlib import Path

from pllm import Model
from pllm.quantization import fit_public_equalization_profile
from pllm.runtime.public_equalization import profile_path


def main() -> None:
    source = os.environ.get("PLLM_REAL_PHI_PATH")
    if not source:
        raise ValueError("PLLM_REAL_PHI_PATH must point to the pinned local Phi-4-mini checkpoint")
    root = Path(source)
    model = Model.path(
        str(root), model_id="microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083",
    )
    # Public offline calibration; not taken from reference-quality prompts or requests.
    cohort = (
        (1, 2, 3), (32, 128, 4096), (42, 384, 4000, 8),
        (8192, 99, 100), (127, 90, 91, 92, 93),
    )
    profile = fit_public_equalization_profile(model, cohort, threads=4)
    path = profile_path(root, profile.digest)
    path.write_bytes(profile.pack())
    print(f"{profile.digest} {len(profile.stage_scales)} stages {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
