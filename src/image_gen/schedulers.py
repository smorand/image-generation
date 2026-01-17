"""Scheduler configurations for SDXL pipeline."""

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
)

SCHEDULER_MAPPING = {
    "dpm++_2m_karras": (
        DPMSolverMultistepScheduler,
        {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
    ),
    "dpm++_2m_sde_karras": (
        DPMSolverMultistepScheduler,
        {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"},
    ),
    "dpm++_sde_karras": (
        DPMSolverSinglestepScheduler,
        {"use_karras_sigmas": True},
    ),
    "euler_a": (
        EulerAncestralDiscreteScheduler,
        {},
    ),
    "euler": (
        EulerDiscreteScheduler,
        {},
    ),
    "ddim": (
        DDIMScheduler,
        {},
    ),
    "unipc": (
        UniPCMultistepScheduler,
        {},
    ),
}

SUPPORTED_SCHEDULERS = list(SCHEDULER_MAPPING.keys())


def get_scheduler(name: str, config: dict) -> object:
    """
    Create a scheduler instance from name and pipeline config.

    Args:
        name: Scheduler name (e.g., 'dpm++_2m_karras')
        config: Scheduler config from pipeline (pipeline.scheduler.config)

    Returns:
        Configured scheduler instance
    """
    if name not in SCHEDULER_MAPPING:
        raise ValueError(
            f"Unknown scheduler: {name}. Supported: {SUPPORTED_SCHEDULERS}"
        )

    scheduler_class, extra_kwargs = SCHEDULER_MAPPING[name]
    return scheduler_class.from_config(config, **extra_kwargs)
