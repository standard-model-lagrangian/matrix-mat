#!/usr/bin/env python3
"""
Forwarding CLI entrypoint for the brightfield spheroid volume pipeline.
Delegates directly to matrix_mat.brightfield.run_pipeline.main (prefer `matrix-mat bf run`).
"""

import sys

from matrix_mat.brightfield.run_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
