"""Populate Nansen's ERA5 surface-pressure cache. See beardmore.cache_climate."""

from __future__ import annotations

import argparse
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

from stereo_melt.pipeline import populate_climate_cache

from nansen import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config.ensure_output_dirs()
    populate_climate_cache(config, cadence_hours=args.cadence, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
