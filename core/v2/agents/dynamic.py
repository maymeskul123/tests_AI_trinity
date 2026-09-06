from __future__ import annotations

from .quaternary import QuaternaryAgent


class DynamicAgent(QuaternaryAgent):
    name = "DYNAMIC"

    # Phase 3 test01 deliberately keeps the same
    # state semantics as QUATERNARY.
    #
    # Adaptive decision thresholds will be introduced
    # only in a later benchmark, so that test01 verifies
    # the logical state model independently.

    pass
