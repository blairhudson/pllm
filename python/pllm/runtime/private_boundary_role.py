"""Standalone loopback launch for one coupled-boundary role."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pllm.configuration import Experiment

from .private_boundary_loader import create_private_boundary_role_app
from .private_boundary_protocol import BoundaryError, ROLES


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("UNSUPPORTED_NETWORK_GRAPH: private boundary supports loopback children only")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        raw = (args.experiment.read_text(encoding="utf-8") if args.experiment else
               os.environ.get("PLLM_ROLE_EXPERIMENT_JSON"))
        if raw is None:
            raise BoundaryError("private boundary requires a bound Experiment")
        experiment = Experiment.from_spec(json.loads(raw))
        app = create_private_boundary_role_app(
            experiment, args.role, os.environ["PLLM_BOUNDARY_API_KEY"],
            os.environ["PLLM_BOUNDARY_PEER_CREDENTIAL_DIGEST"],
            cache_dir=os.environ.get("HF_HUB_CACHE"),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
