#!/usr/bin/env python3
"""
Forwarding CLI entrypoint for Spheroid Volume Analysis Pipeline v2.
Delegates directly to spheroid_pipeline_v2.run_pipeline.
"""

import sys
from spheroid_pipeline_v2.run_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
