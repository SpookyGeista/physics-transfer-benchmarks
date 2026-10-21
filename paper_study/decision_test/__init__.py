"""Prospectively locked held-out extension of the unchanged scalar decision study."""
import os

# Before importing StateID/Numba: old studies also imported the same source as
# unqualified ``state_id``. Its pickle cache cannot be shared with the qualified
# package import used here. Keep an explicit caller-selected cache if provided.
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/physics-transfer-decision-test-v1-qualified-numba')
