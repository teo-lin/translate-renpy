"""
Aya-23-8B translator — thin subclass of LlamaCppTranslator with Aya-23 defaults.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
from translators.llama_cpp_translator import LlamaCppTranslator


def _get_default_model_path():
    """Download Aya-23-8B from HF cache and return local blob path."""
    return hf_hub_download(
        repo_id="bartowski/aya-23-8B-GGUF",
        filename="aya-23-8B-Q4_K_M.gguf",
        local_files_only=False
    )


class Aya23Translator(LlamaCppTranslator):
    """LlamaCppTranslator pre-configured for Aya-23-8B with its original defaults."""

    def __init__(
        self,
        model_path: str = None,
        target_language: str = "Romanian",
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        n_batch: int = 256,
        prompt_template: str = None,
        glossary: dict = None,
    ):
        if model_path is None:
            model_path = _get_default_model_path()
        super().__init__(
            model_path=str(model_path),
            target_language=target_language,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            prompt_template=prompt_template,
            glossary=glossary,
        )
