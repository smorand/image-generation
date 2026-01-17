"""Prompt encoding with compel for long prompts and weighting support."""

import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

# Suppress warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("compel").setLevel(logging.ERROR)

import torch

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline


@contextmanager
def suppress_output():
    """Suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SDXLPromptEncoder:
    """
    SDXL prompt encoder using compel.

    Supports:
    - Long prompts (>77 tokens) via automatic chunking
    - Prompt weighting syntax: (word:1.2), word+, word-
    - Blending prompts: "cat AND dog"
    """

    def __init__(self, pipeline: "StableDiffusionXLPipeline"):
        """
        Initialize the prompt encoder.

        Args:
            pipeline: Loaded SDXL pipeline
        """
        from compel import Compel, ReturnedEmbeddingsType

        self.pipeline = pipeline

        # Suppress compel deprecation warning (printed to stderr)
        with suppress_output():
            self.compel = Compel(
                tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,  # Enable long prompt support
            )

    def _pad_embeddings(
        self,
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad embeddings to have the same sequence length.

        Args:
            prompt_embeds: Positive prompt embeddings
            negative_embeds: Negative prompt embeddings

        Returns:
            Tuple of padded (prompt_embeds, negative_embeds)
        """
        prompt_len = prompt_embeds.shape[1]
        negative_len = negative_embeds.shape[1]

        if prompt_len == negative_len:
            return prompt_embeds, negative_embeds

        max_len = max(prompt_len, negative_len)

        if prompt_len < max_len:
            # Pad positive embeddings
            padding = prompt_embeds[:, -1:, :].repeat(1, max_len - prompt_len, 1)
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        else:
            # Pad negative embeddings
            padding = negative_embeds[:, -1:, :].repeat(1, max_len - negative_len, 1)
            negative_embeds = torch.cat([negative_embeds, padding], dim=1)

        return prompt_embeds, negative_embeds

    def encode(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode prompts into embeddings.

        Args:
            prompt: Positive prompt (supports weighting syntax)
            negative_prompt: Negative prompt (supports weighting syntax)

        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        # Encode prompts (suppress token length warnings printed to stderr)
        with suppress_output():
            conditioning, pooled = self.compel(prompt)

            if negative_prompt:
                negative_conditioning, negative_pooled = self.compel(negative_prompt)
            else:
                negative_conditioning, negative_pooled = self.compel("")

        # Pad embeddings to same length (required by diffusers pipeline)
        conditioning, negative_conditioning = self._pad_embeddings(
            conditioning, negative_conditioning
        )

        return conditioning, negative_conditioning, pooled, negative_pooled

    def get_embeddings_for_pipeline(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> dict:
        """
        Get embeddings formatted for pipeline call.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt

        Returns:
            Dict with prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        """
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode(prompt, negative_prompt)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
