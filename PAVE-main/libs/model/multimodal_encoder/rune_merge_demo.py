import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from libs.model.multimodal_encoder.siglip_encoder import SigLipVisionTower


def hook_k(module, input, output):
    outputs['desired_k'] = output


def hook_q(module, input, output):
    outputs['desired_q'] = output


outputs = {}


def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1,)
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()
    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    ratio = len(np.where((attn_np > upper_bound))[0]) / len(attn_np)
    return ratio


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)


def visualize_attention_distribution(cls_attn, title="CLS Token Attention Distribution"):
    plt.figure(figsize=(10, 5))
    plt.hist(cls_attn.cpu().numpy().flatten(), bins=50, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def main():
    # 初始化模型
    model_name = "google/siglip2-so400m-patch14-384"
    vision_tower = SigLipVisionTower(model_name, None, delay_load=False)

    # 加载图像
    image_path = "path/to/your/image.jpg"  # 替换为你的图像路径
    image_tensor = load_image(image_path)

    # 提取原始 patch embeddings
    with torch.no_grad():
        # 注册钩子获取 k 和 q
        hook_handle_k = vision_tower.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = vision_tower.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        image_forward_outs = vision_tower.vision_tower(image_tensor.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
        cls_token_last_layer = image_forward_outs.hidden_states[-2][:, 0:1]
        image_features = image_forward_outs.hidden_states[-2].to(image_tensor.dtype)
        B, N, C = image_features.shape

        # 获取 attention 权重
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]

        # 自适应剪枝比例
        reduction_ratio = outlier_dectection(cls_attn)

        _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)
        index = idx.unsqueeze(-1).expand(-1, -1, C)

        Key_wo_cls = desired_layer_k[:, 1:]
        x_others = torch.gather(image_features, dim=1, index=index)
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)

        compl = complement_idx(idx, N)
        non_topk = torch.gather(image_features, dim=1,
                                index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_Key = torch.gather(Key_wo_cls, dim=1,
                                    index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

        Key_others_norm = F.normalize(x_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b, i, :].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)
                after_i_Key = Key_others_norm[b, i + 1:, :].unsqueeze(0)

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)
                after_i_x_others = x_others[b, i + 1:, :].unsqueeze(0)
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b, :, :].unsqueeze(0)], dim=1)

                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)
                after_i_x_others_attn = x_others_attn[b, i + 1:].unsqueeze(0)
                rest_x_others_attn = torch.cat(
                    [before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b, :].unsqueeze(0)], dim=1
                )

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b, :, :].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=32, dim=2, largest=True)

                cluster_tokens = rest_x_others[:, cluster_indices.squeeze(), :]
                weights = rest_x_others_attn[:, cluster_indices.squeeze()].unsqueeze(-1)

                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
                updated_center = x_others[b, i, :] + weighted_avg
                updated_x_others[b, i, :] = updated_center

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
        updated_x_others = torch.cat([updated_x_others, extra_one_token], dim=1)
        pruned_features = updated_x_others

        print(f"Original features shape: {image_features.shape}")
        print(f"Pruned features shape: {pruned_features.shape}")
        print(f"Retained tokens count: {pruned_features.shape[1]} out of {image_features.shape[1]}")

        visualize_attention_distribution(cls_attn)


if __name__ == "__main__":
    main()