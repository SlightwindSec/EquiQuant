import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.common import cleanup_memory
from ...utils.logger import logger
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeTopKRouter,
)


class Qwen3_5MoeSparseSplitMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = nn.ModuleList(
            [Qwen3_5MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        expert_output = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            expert_output.index_add_(0, token_idx, current_hidden_states.to(expert_output.dtype))

        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


def convert_experts_to_mlp(
    original_moe_block,
    config,
) -> Qwen3_5MoeSparseSplitMoeBlock:
    new_moe_block = Qwen3_5MoeSparseSplitMoeBlock(config)

    with torch.no_grad():
        new_moe_block.gate.weight.copy_(original_moe_block.gate.weight)

        for expert_idx in range(config.num_experts):
            gate_up_weight = original_moe_block.experts.gate_up_proj[expert_idx]
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
            new_moe_block.experts[expert_idx].gate_proj.weight.copy_(gate_weight)
            new_moe_block.experts[expert_idx].up_proj.weight.copy_(up_weight)
            new_moe_block.experts[expert_idx].down_proj.weight.copy_(
                original_moe_block.experts.down_proj[expert_idx]
            )

        new_moe_block.shared_expert.gate_proj.weight.copy_(original_moe_block.shared_expert.gate_proj.weight)
        new_moe_block.shared_expert.up_proj.weight.copy_(original_moe_block.shared_expert.up_proj.weight)
        new_moe_block.shared_expert.down_proj.weight.copy_(original_moe_block.shared_expert.down_proj.weight)
        new_moe_block.shared_expert_gate.weight.copy_(original_moe_block.shared_expert_gate.weight)

    new_moe_block = new_moe_block.to(
        device=original_moe_block.gate.weight.device,
        dtype=original_moe_block.gate.weight.dtype,
    )
    return new_moe_block


def convert_qwen3_5_moe(model):
    logger.info("Converting experts to mlp for qwen3_5_moe...")
    for i, layer in enumerate(model.layers[: model.config.num_hidden_layers]):
        original_mlp = layer.mlp
        layer.mlp = convert_experts_to_mlp(original_mlp, model.config)
        del original_mlp
    
    cleanup_memory()
