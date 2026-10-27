"""Explicit hydrogen-engine experiment settings; original CPU call is unchanged."""


def hydrogen_properties(policy):
    if policy == "cpu1":
        return "CPU", None
    values = {"gpu_single": "single", "gpu_mixed": "mixed", "gpu_double": "double"}
    if policy not in values:
        raise ValueError(f"Unknown hydrogen engine: {policy}")
    return "CUDA", {"Precision": values[policy], "DeterministicForces": "true"}
