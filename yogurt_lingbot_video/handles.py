from dataclasses import dataclass
from typing import Any


LINGBOT_VIDEO_PIPELINE_TYPE = "YOGURT_LINGBOT_VIDEO_PIPELINE"
LINGBOT_VIDEO_REWRITER_TYPE = "YOGURT_LINGBOT_VIDEO_REWRITER"


@dataclass
class LingBotVideoPipelineHandle:
    pipeline: Any
    model_path: str
    mode: str
    device: str
    transformer_dtype: str
    cpu_offload: bool
    moe_sequential_cpu_offload: bool = False
    moe_block_swap: bool = False

    @property
    def dtype(self) -> str:
        return self.transformer_dtype


@dataclass
class LingBotVideoRewriterHandle:
    backend: Any
    base_model_path: str
    adapter_model_path: str
