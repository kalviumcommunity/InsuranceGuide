"""
Basic token and approximate cost tracking for RAG requests.
"""

import os
from typing import Any


# Approximate Gemini input/output prices per 1M tokens.
# These values are estimates and can be updated when pricing changes.
INPUT_COST_PER_MILLION = float(
    os.getenv(
        "INPUT_COST_PER_MILLION",
        "0.10",
    )
)

OUTPUT_COST_PER_MILLION = float(
    os.getenv(
        "OUTPUT_COST_PER_MILLION",
        "0.40",
    )
)


def estimate_token_count(text: str) -> int:
    """
    Estimate token usage when exact model usage data
    is not available.

    A simple estimate of 1 token per 4 characters is used.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // 4,
    )


def calculate_usage(
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    """
    Calculate approximate input/output tokens
    and estimated request cost.
    """

    input_tokens = estimate_token_count(
        prompt
    )

    output_tokens = estimate_token_count(
        answer
    )

    estimated_cost = (
        (
            input_tokens
            / 1_000_000
        )
        * INPUT_COST_PER_MILLION
    ) + (
        (
            output_tokens
            / 1_000_000
        )
        * OUTPUT_COST_PER_MILLION
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens
            + output_tokens
        ),
        "estimated_cost": round(
            estimated_cost,
            8,
        ),
    }
