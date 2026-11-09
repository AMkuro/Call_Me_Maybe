"""Prompt text sent to the model."""

from .schema import FunctionDef


def build_name_prompt(
    user_prompt: str, functions: list[FunctionDef]
) -> str:
    """Prompt for choosing the function name."""
    raise NotImplementedError


def build_parameters_prompt(user_prompt: str, fn: FunctionDef) -> str:
    """Prompt for generating the parameter values of ``fn``."""
    raise NotImplementedError
