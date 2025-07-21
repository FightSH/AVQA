import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# from visual_net import resnet18

from ipdb import set_trace
import timm
from torch.distributions.normal import Normal
from einops import rearrange, repeat
from audio_layers import Transformer_Layer
from audio_others import SparseDispatcher, series_decomp_multi, MLP


class AMS(nn.Module):
    def __init__(self, input_size, output_size, seq_lenth, num_experts, device, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1,
                 batch_norm=False):
        super(AMS, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.seq_lenth = seq_lenth
        self.k = k

        self.start_linear = nn.Linear(in_features=input_size, out_features=1)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(self.seq_lenth / patch)
            self.experts.append(Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff,
                                                  dynamic=dynamic, patch_nums=patch_nums,
                                                  patch_size=patch, factorized=True, layer_number=layer_number,
                                                  batch_norm=batch_norm))

        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Linear(input_size, num_experts)
        self.w_gate = nn.Linear(input_size, num_experts)

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def trend_decompose(self, x):
        # x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        return x + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        # x = self.start_linear(x).squeeze(-1)

        print(f"noisy_top_k_gating x shape: {x.shape}")
        # 直接使用输入 x，需要将其重塑为门控网络期望的形状
        # x = x.view(x.size(0), -1)
        x = x.mean(dim=1)

        # clean_logits = x @ self.w_gate
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, v_q, loss_coef=1e-2):
        new_x = self.trend_decompose(x)
        print(f"new_x shape: {new_x.shape}")
        # multi-scale router
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        print(f"gates shape: {gates.shape}, load shape: {load.shape}")
        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(new_x)  # 分发输入到各个专家
        expert_v_q = dispatcher.dispatch(v_q)  # 新增：对v_q进行分发
        print(f"expert_inputs shape: {[inp.shape for inp in expert_inputs]}")
        print(f"expert_v_q shape: {[vq.shape for vq in expert_v_q]}")
        expert_outputs = [self.experts[i](expert_inputs[i], expert_v_q[i])[0] for i in range(self.num_experts)]
        print(f"expert_outputs shape: {[out.shape for out in expert_outputs]}")
        output = dispatcher.combine(expert_outputs)
        print(f"output shape after combine: {output.shape}")
        if self.residual_connection:
            output = output + x
        return output, balance_loss
