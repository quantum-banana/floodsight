"""Audited SegFormer-B2 and offline synthetic model construction."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor
from torch.nn import functional as F

from .artifact import ModelArtifact
from .config import ModelConfig
from .errors import ArtifactError, DependencyError


def _transformers() -> tuple[type, type]:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise DependencyError(
            "Transformers is required; activate the pinned FloodSight ML environment."
        ) from exc
    return SegformerConfig, SegformerForSemanticSegmentation


def _b2_config(config_type: type, config: ModelConfig):
    """Construct the audited B2 architecture without reading remote/local config code."""

    id2label = dict(enumerate(config.class_names))
    return config_type(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=768,
        semantic_loss_ignore_index=255,
        reshape_last_stage=True,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id={name: class_id for class_id, name in id2label.items()},
    )


def _loading_key(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return value[0]
    return repr(value)


def build_segformer(config: ModelConfig, artifact: ModelArtifact):
    """Load the frozen B2 baseline from one already-hashed local safetensors file."""

    config_type, model_type = _transformers()
    if not config.local_files_only or config.trust_remote_code:
        raise ArtifactError("Network fallback and remote model code are forbidden.")
    if artifact.safetensors_path.name != "model.safetensors":
        raise ArtifactError("Validated weights must be named model.safetensors.")
    model, loading_info = model_type.from_pretrained(
        str(artifact.safetensors_path.parent),
        config=_b2_config(config_type, config),
        ignore_mismatched_sizes=True,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    allowed_head_keys = {"decode_head.classifier.weight", "decode_head.classifier.bias"}
    missing = {_loading_key(item) for item in loading_info.get("missing_keys", [])}
    unexpected = {_loading_key(item) for item in loading_info.get("unexpected_keys", [])}
    mismatched = {_loading_key(item) for item in loading_info.get("mismatched_keys", [])}
    errors = loading_info.get("error_msgs", [])
    required_head_keys = allowed_head_keys
    observed_head_keys = missing | mismatched
    incompatible = (
        unexpected
        or errors
        or not missing <= allowed_head_keys
        or not mismatched <= allowed_head_keys
        or observed_head_keys != required_head_keys
    )
    if incompatible:
        raise ArtifactError(
            "Local SegFormer weights are incompatible with the audited B2 architecture: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"mismatched={sorted(mismatched)}, errors={errors}."
        )
    return model


def build_tiny_offline_segformer(*, num_labels: int, class_names: Sequence[str] | None = None):
    """Build a tiny random SegFormer for synthetic tests without network access."""

    config_type, model_type = _transformers()
    names = tuple(class_names or (f"class_{index}" for index in range(num_labels)))
    if len(names) != num_labels:
        raise ValueError("class_names must contain num_labels entries.")
    config = config_type(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[8, 16, 32, 64],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 4, 8],
        mlp_ratios=[2, 2, 2, 2],
        decoder_hidden_size=32,
        num_labels=num_labels,
        id2label=dict(enumerate(names)),
        label2id={name: index for index, name in enumerate(names)},
    )
    return model_type(config)


def logits_at_label_resolution(logits: Tensor, labels: Tensor) -> Tensor:
    """Bilinearly resize model logits to the target-mask resolution."""

    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,H,W].")
    if logits.shape[-2:] == labels.shape[-2:]:
        return logits
    return F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
