"""
Transformers Backend for Local LLM Inference with Attention Extraction.

Supports Qwen2.5 and other models with attention extraction for semantic slicing.
Optimized for Mac M1/M2 with MPS (Metal Performance Shaders) acceleration.

Example:
    backend = TransformersBackend(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",  # Automatically detect MPS
    )
    result = backend.forward("Hello, world!")
    print(result.attention_weights.shape)  # (layers, heads, seq_len, seq_len)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from agentos.memory.slicing.types import AttentionOutput

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_warning()


class DeviceType(str, Enum):
    """Available device types."""

    AUTO = "auto"  # Automatically detect best device
    CPU = "cpu"  # CPU only
    CUDA = "cuda"  # NVIDIA CUDA
    MPS = "mps"  # Apple Metal Performance Shaders (M1/M2/M3)
    METAL = "metal"  # Alias for MPS


@dataclass
class BackendConfig:
    """Configuration for transformers backend."""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: DeviceType = DeviceType.AUTO
    torch_dtype: str = "auto"  # "auto", "float32", "float16", "bfloat16"

    # Inference configuration
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False

    # Attention output configuration
    output_attentions: bool = True
    output_hidden_states: bool = True

    # Tokenizer configuration
    trust_remote_code: bool = True  # Required for some models like Qwen

    # Memory optimization
    use_cache: bool = True
    low_cpu_mem_usage: bool = True

    # Optional: model revision
    revision: str | None = None

    def validate(self) -> None:
        """Validate configuration."""
        valid_dtypes = {"auto", "float32", "float16", "bfloat16"}
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid torch_dtype: {self.torch_dtype}. "
                f"Must be one of {valid_dtypes}"
            )


class TransformersBackend:
    """Local LLM backend with attention extraction.

    Provides access to attention weights and hidden states for
    semantic slicing and other AgentOS operations.
    """

    def __init__(self, config: BackendConfig | None = None) -> None:
        """Initialize the transformers backend.

        Args:
            config: Backend configuration. If None, uses defaults.
        """
        self.config = config or BackendConfig()
        self.config.validate()

        self._device: torch.device = self._detect_device()
        self._dtype: torch.dtype = self._resolve_dtype()

        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

        logger.info(f"Initialized backend with device: {self._device}")

    def _detect_device(self) -> torch.device:
        """Detect the best available device.

        Returns:
            torch.device to use for inference.
        """
        if self.config.device == DeviceType.AUTO:
            # Try MPS (Mac), then CUDA, then CPU
            if torch.backends.mps.is_available():
                logger.info("Using MPS (Metal Performance Shaders) acceleration")
                return torch.device("mps")
            elif torch.cuda.is_available():
                logger.info("Using CUDA acceleration")
                return torch.device("cuda")
            else:
                logger.info("Using CPU (no accelerator detected)")
                return torch.device("cpu")
        else:
            return torch.device(self.config.device.value)

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve torch dtype from config string.

        Returns:
            torch.dtype to use for model weights.
        """
        if self.config.torch_dtype == "auto":
            # Use float16 on GPU, float32 on CPU/MPS
            if self._device.type == "cuda":
                return torch.float16
            else:
                return torch.float32
        elif self.config.torch_dtype == "float16":
            return torch.float16
        elif self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    def load(self) -> None:
        """Load the model and tokenizer.

        This is separate from __init__ to allow lazy loading.
        """
        if self._model is not None:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
        )

        # Ensure pad token is set (some models don't have it)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with attention output enabled
        # CRITICAL: output_attentions must be set during model loading,
        # not just in forward(), for models that support it
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=self._dtype,
            device_map=self._device.type if self._device.type != "mps" else None,
            trust_remote_code=self.config.trust_remote_code,
            output_attentions=True,  # Enable attention output during loading
            output_hidden_states=True,  # Enable hidden states during loading
            use_cache=self.config.use_cache,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            revision=self.config.revision,
        )

        # Move to device for MPS (can't use device_map)
        if self._device.type == "mps":
            self._model = self._model.to(self._device)

        self._model.eval()

        logger.info(f"Model loaded on {self._device} with dtype {self._dtype}")

    @property
    def model(self) -> AutoModelForCausalLM:
        """Get the model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Get the device being used."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype being used."""
        return self._dtype

    def tokenize(self, text: str) -> tuple[list[str], torch.Tensor]:
        """Tokenize text.

        Args:
            text: Input text to tokenize.

        Returns:
            Tuple of (tokens as strings, token IDs as tensor).
        """
        token_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        # Convert to tokens for debugging
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids[0])

        return tokens, token_ids

    def forward(
        self,
        text: str,
        max_length: int | None = None,
    ) -> AttentionOutput:
        """Run a forward pass and extract attention weights.

        Args:
            text: Input text.
            max_length: Maximum sequence length. If None, uses config default.

        Returns:
            AttentionOutput with tokens, hidden states, and attention weights.
        """
        if max_length is None:
            max_length = self.config.max_length

        # Tokenize
        tokens, token_ids = self.tokenize(text)
        token_ids = token_ids[:, :max_length]  # Truncate if needed

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(token_ids)

        # Forward pass with attention and hidden states output
        with torch.no_grad():
            outputs = self.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        # Fail fast if model doesn't support attention extraction
        if outputs.attentions is None:
            raise RuntimeError(
                f"Model '{self.config.model_name}' does not output attention weights. "
                "Attention extraction is required for semantic slicing. "
                "Try a different model or ensure the model supports attention output."
            )
        if outputs.hidden_states is None:
            raise RuntimeError(
                f"Model '{self.config.model_name}' does not output hidden states. "
                "Hidden states are required for semantic slicing."
            )

        # Extract hidden states
        # Shape: (num_layers, batch_size, seq_len, hidden_dim)
        hidden_states = torch.stack(outputs.hidden_states)
        # Remove batch dimension: (num_layers, seq_len, hidden_dim)
        hidden_states = hidden_states.squeeze(1)
        # Use last layer hidden state: (seq_len, hidden_dim)
        last_hidden_state = hidden_states[-1].cpu().float()

        # Extract attention weights
        # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.stack(outputs.attentions)
        # Remove batch dimension: (num_layers, num_heads, seq_len, seq_len)
        attention_weights = attention_weights.squeeze(1)
        # Convert to numpy on CPU: (num_layers, num_heads, seq_len, seq_len)
        attention_weights = attention_weights.cpu().float().numpy()

        # Get tokens list and decoded text
        tokens_list = [str(t) for t in tokens[: token_ids.shape[1]]]
        token_ids_list = token_ids[0].cpu().tolist()

        # Decode the full text properly (handles spaces, special tokens, etc.)
        decoded_text = self.tokenizer.decode(token_ids_list, skip_special_tokens=True)

        return AttentionOutput(
            tokens=tokens_list,
            token_ids=token_ids_list,
            decoded_text=decoded_text,
            hidden_states=last_hidden_state.numpy(),
            attention_weights=attention_weights,
            metadata={
                "model_name": self.config.model_name,
                "device": str(self._device),
                "dtype": str(self._dtype),
                "num_layers": attention_weights.shape[0],
                "num_heads": attention_weights.shape[1],
                "seq_len": attention_weights.shape[2],
            },
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            system_prompt: Optional system prompt for chat models.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text.
        """
        # Base generation config (only include supported parameters)
        # Use max_new_tokens for generation length control
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            # Explicitly disable these for generation (they're for model.forward only)
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
        }

        # Only add additional kwargs that are supported by generate()
        # Filter out unsupported parameters like top_k, top_k=None to avoid warnings
        unsupported_params = {
            "top_k",  # Not supported by all models
            "output_attentions",  # Not for generate
            "output_hidden_states",  # Not for generate
            "output_scores",  # Not for generate
        }

        for key, value in kwargs.items():
            if key not in unsupported_params and value is not None:
                generation_config[key] = value

        # Tokenize with attention mask
        # Use chat template for models that support it (Qwen, LLaMA, etc.)
        if system_prompt and hasattr(self.tokenizer, 'apply_chat_template'):
            # Format as chat message with system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        else:
            # Standard tokenization for non-chat models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Track input length for proper prompt removal
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **generation_config,
            )

        # Decode only the generated tokens (skip input tokens)
        # outputs[0] contains [input_tokens + generated_tokens]
        # We only decode tokens starting from index input_length
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Clear cache
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            # MPS doesn't have empty_cache, but we can trigger garbage collection
            import gc

            gc.collect()

        logger.info("Model unloaded")

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        _ = exc_type, exc_val, exc_tb  # Unused but required by protocol
        self.unload()
        return False


def create_backend(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: DeviceType = DeviceType.AUTO,
    **kwargs,
) -> TransformersBackend:
    """Convenience function to create a transformers backend.

    Args:
        model_name: Model name or path.
        device: Device to use.
        **kwargs: Additional configuration options.

    Returns:
        Initialized TransformersBackend.
    """
    config = BackendConfig(model_name=model_name, device=device, **kwargs)
    return TransformersBackend(config)
