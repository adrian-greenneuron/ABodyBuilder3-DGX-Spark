"""Explicit experiment settings; no global OpenMM properties are modified."""


def repair_properties(policy):
    if policy == "reference":
        return None, None
    if policy == "cuda_deterministic":
        return "CUDA", {"Precision": "single", "DeterministicForces": "true"}
    if policy == "cpu1":
        return "CPU", {"Threads": "1", "DeterministicForces": "true"}
    if policy == "cuda_double":
        return "CUDA", {"Precision": "double", "DeterministicForces": "true"}
    raise ValueError(f"Unknown experimental repair policy: {policy}")
