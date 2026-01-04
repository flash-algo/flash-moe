import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from flash_moe.ops.triton.flash_mlp import triton_flash_mlp_func
from flash_moe.ops.triton.flash_router import triton_flash_router_func
from flash_moe.ops.triton.flash_expert import triton_flash_expert_func


class FlashMoEConfig:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = False,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob


class FlashMoE(nn.Module):
    def __init__(self, config: FlashMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.num_experts = config.num_experts
        self.num_keys = math.floor(math.sqrt(self.num_experts))
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # shared expert
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # router gate for retrieval experts
        self.router_gate = nn.Linear(self.hidden_size, self.num_keys * 2, bias=False)
        self.router_norm = nn.BatchNorm1d(self.hidden_size, affine=False)

        # routed experts
        self.down_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.up_embed = nn.Embedding(self.num_experts, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        # get routing logits with router gate
        router_logits = self.router_gate(self.router_norm(hidden_states))
        router_logits = router_logits.view(bsz * seq_len, 2, -1).transpose(0, 1)

        # Calculate log probabilities for routing
        # We use log_softmax because for product keys
        # log(P(x, y)) = log(P(x)) + log(P(y))
        # This allows us to use the efficient addition structure while working with probabilities
        router_log_probs = F.log_softmax(router_logits, dim=-1)

        # get experts with the highest routing probabilities
        scores, indices = triton_flash_router_func(
            router_log_probs, self.num_keys, self.top_k
        )

        # Convert log-probabilities back to probabilities
        routing_weights = torch.exp(scores)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )

        # mix routed experts states with shared expert states
        experts_states = triton_flash_expert_func(
            hidden_states,
            self.down_embed.weight,
            self.up_embed.weight,
            indices,
            routing_weights,
        )
        hidden_states = triton_flash_mlp_func(
            hidden_states,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        )
        hidden_states = hidden_states + experts_states

        hidden_states = hidden_states.view(bsz, seq_len, -1)

        return hidden_states, router_logits


class FlashMoE_Pytorch(nn.Module):
    def __init__(self, config: FlashMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.num_experts = config.num_experts
        self.num_keys = math.floor(math.sqrt(self.num_experts))
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # shared expert
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )

        # router gate for retrieval experts
        self.router_gate = nn.Linear(self.hidden_size, self.num_keys * 2, bias=False)
        self.router_norm = nn.BatchNorm1d(self.hidden_size, affine=False)

        # routed experts
        self.down_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.up_embed = nn.Embedding(self.num_experts, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # get routing logits with router gate
        router_logits = self.router_gate(
            self.router_norm(hidden_states.view(-1, self.hidden_size))
        )
        router_logits = router_logits.view(bsz * seq_len, 2, -1).transpose(0, 1)

        # Calculate log probabilities for routing
        # We use log_softmax because for Product Keys, P(x, y) = P(x) * P(y)
        # log(P(x, y)) = log(P(x)) + log(P(y))
        # This allows us to use the efficient addition structure while working with probabilities
        router_log_probs = F.log_softmax(router_logits, dim=-1)

        # get experts with the highest routing probabilities
        (scores_x, scores_y), (indices_x, indices_y) = router_log_probs.topk(
            self.num_keys, dim=-1
        )
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)
        all_scores = all_scores.view(*all_scores.shape[:-2], -1)
        all_indices = all_indices.view(*all_indices.shape[:-2], -1)
        scores, position_indices = all_scores.topk(self.top_k, dim=-1)
        indices = all_indices.gather(-1, position_indices)

        # Convert log-probabilities back to probabilities
        routing_weights = torch.exp(scores)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )

        # mix routed experts states with shared expert states
        down_embed = self.down_embed(indices)
        up_embed = self.up_embed(indices)
        experts_weights = torch.matmul(
            down_embed, hidden_states.view(bsz * seq_len, -1, 1)
        ).view(bsz * seq_len, -1)
        experts_weights = self.act_fn(experts_weights) * routing_weights
        experts_states = torch.matmul(
            experts_weights.view(bsz * seq_len, 1, -1), up_embed
        ).view(bsz, seq_len, -1)
        hidden_states = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        hidden_states = hidden_states + experts_states
        return hidden_states, router_logits
