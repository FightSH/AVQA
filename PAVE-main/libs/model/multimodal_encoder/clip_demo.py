import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoProcessor
import numpy as np

class TokenMergeFeatureExtractor:
    def __init__(self, vision_tower_name, select_layer=-2, device='cuda'):
        self.vision_tower_name = vision_tower_name
        self.select_layer = select_layer
        self.device = device

        # self.processor = AutoProcessor.from_pretrained(vision_tower_name,cache_dir='/mnt/sda/shenhao/models')
        # 加载模型
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name,cache_dir='/mnt/sda/shenhao/models')
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name,cache_dir='/mnt/sda/shenhao/models')
        self.vision_tower.requires_grad_(False)
        self.vision_tower.to(device)
        self.dtype = self.vision_tower.dtype
        self.select_feature = 'patch'

        # 用于存储hook输出
        self.outputs = {}

    def hook_k(self, module, input, output):
        self.outputs['desired_k'] = output

    def hook_q(self, module, input, output):
        self.outputs['desired_q'] = output

    def complement_idx(self, idx, dim):
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

    def outlier_detection(self, attn):
        attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()
        Q1 = np.percentile(attn_np, 25)
        Q3 = np.percentile(attn_np, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = np.where((attn_np > upper_bound))[0]
        ratio = len(outlier_indices) / len(attn_np)
        return ratio

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]  # penultimate layer output
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def extract_features(self, images, method='advanced', if_adaptive=True, reduction_ratio=1 / 8):
        """
        提取特征的主要方法
        Args:
            images: 输入图像张量 [B, C, H, W]
            method: 'advanced' 或 'advanced_plus'
            if_adaptive: 是否使用自适应缩减比例
            reduction_ratio: 缩减比例
        """
        with torch.no_grad():
            if method == 'advanced':
                return self._token_prune_merge_advanced(images, if_adaptive, reduction_ratio)
            elif method == 'advanced_plus':
                return self._token_prune_merge_advanced_plus(images, if_adaptive, reduction_ratio)
            else:
                raise ValueError(f"Unknown method: {method}")

    def _token_prune_merge_advanced(self, images, if_adaptive=True, reduction_ratio=1 / 8):
        '''
        version 10/03/2024 using the key*key matrix to calculate the cosine similarity
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        # set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(self.hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(self.hook_q)

        # forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        cls_token_last_layer = image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        # extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = self.outputs["desired_k"]
        desired_layer_q = self.outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]

        if if_adaptive:
            reduction_ratio = self.outlier_detection(cls_attn)  # *3.5
        _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = self.complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1,
                                index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

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
                    [before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b, :].unsqueeze(0)], dim=1)

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b, :, :].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:, cluster_indices.squeeze(), :]
                weights = rest_x_others_attn[:, cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)  # / torch.sum(weights)
                updated_center = weighted_avg + x_others[b, i, :]
                updated_x_others[b, i, :] = updated_center

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token], dim=1)
        image_features = updated_x_others
        return image_features

    def _token_prune_merge_advanced_plus(self, images, if_adaptive=True, reduction_ratio=1 / 8):
        # 注册hooks
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(
            self.hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(
            self.hook_q)

        # 前向传播
        image_forward_outs = self.vision_tower(images.to(device=self.device), output_hidden_states=True)
        cls_token_last_layer = image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape
        # print(image_features.shape)
        # 提取k和q并移除hooks
        desired_layer_k = self.outputs["desired_k"]
        desired_layer_q = self.outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        # 计算注意力
        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)
        cls_attn = attn[:, 0, 1:]

        if if_adaptive:
            original_reduction_ratio = reduction_ratio
            reduction_ratio = self.outlier_detection(cls_attn)
            # print(f"原始缩减比例: {original_reduction_ratio}, 自适应后: {reduction_ratio}")

        topk_tokens = int(N * reduction_ratio)
        # print(f"TopK选择的token数量: {topk_tokens}")

        _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)

        # 添加空间采样的tokens
        if if_adaptive:
            step_length = int(1 / reduction_ratio)
            arithmetic_sequence = torch.arange(0, 575, int(step_length / 3)).to(device=self.device)
            original_tensor_1d = idx.flatten().to(device=self.device)
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(
                device=self.device)
            concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
            idx = concatenated_tensor
            # # print("idx_new: ", idx)
        else:
            # # this is for training
            step_length = int(1 / reduction_ratio)
            new_idx = torch.zeros((idx.size(0), idx.size(1) * 2), dtype=torch.long).to(device=self.device)
            for i in range(idx.size(0)):
                arithmetic_sequence = torch.arange(int(step_length / 2), 575, int(step_length)).to(device=self.device)
                original_tensor_1d = idx[i].flatten().to(device=self.device)
                filtered_sequence = arithmetic_sequence
                # filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
                concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
                new_idx[i] = concatenated_tensor
            idx = new_idx

        # print(f"TopK tokens: {len(original_tensor_1d)}")
        # print(f"空间采样补充: {len(filtered_sequence)}")
        # print(f"总token数量: {len(original_tensor_1d) + len(filtered_sequence)}")


        index = idx.unsqueeze(-1).expand(-1, -1, C)
        Key_wo_cls = desired_layer_k[:, 1:]

        x_others = torch.gather(image_features, dim=1, index=index)
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)

        compl = self.complement_idx(idx, N)
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        # Token merging逻辑
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
                    [before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b, :].unsqueeze(0)], dim=1)

                rest_Keys = torch.cat([before_i_Key, after_i_Key], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=min(32, rest_Keys.size(1)), dim=2, largest=True)

                cluster_tokens = rest_x_others[:, cluster_indices.squeeze(), :]
                weights = rest_x_others_attn[:, cluster_indices.squeeze()].unsqueeze(-1)

                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
                updated_center = x_others[b, i, :] + weighted_avg
                updated_x_others[b, i, :] = updated_center

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
        updated_x_others = torch.cat([updated_x_others, extra_one_token], dim=1)

        return updated_x_others


# 初始化特征提取器
extractor = TokenMergeFeatureExtractor(
    vision_tower_name="openai/clip-vit-large-patch14-336",
    device='cuda:2' if torch.cuda.is_available() else 'cpu'
)


# 预处理图像并提取特征
def extract_image_features(image_path):
    from PIL import Image

    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')

    # inputs = extractor.processor(images=image,
    #                     return_tensors="pt",crop_size={"height": 336, "width": 336})


    inputs = extractor.image_processor(image, return_tensors="pt")


    # advanced
    # TRUE 1, 116, 1024 0.45MB
    # FALSE 1, 73, 1024 0.29MB
    # advanced plus
    # TRUE 1, 576, 1024 2.25
    # FALSE 1, 145, 1024 0.57

    # print(f"输入图像形状: {inputs['pixel_values'].shape}")
    # 提取特征
    features = extractor.extract_features(
        inputs['pixel_values'],
        method='advanced_plus',
        if_adaptive=False,
        reduction_ratio=1 / 8
    )

    return features


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):
    """
    从视频帧中提取CLIP图像特征，并保存为npy文件

    参数:
        dir_fps_path (str): 包含视频帧的目录路径
        dst_clip_path (str): 特征保存的目标目录路径
    """
    # 获取视频列表
    global img_features
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)

        # 构造保存文件路径
        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        # 获取视频帧列表
        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))

        img_features = torch.zeros(len(video_img_list), 145, 1024)

        idx = 0
        for img_cont in video_img_list:
            # 提取单帧特征
            img_idx_feat = extract_image_features(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        # 将特征转换为numpy数组并保存
        img_features = img_features.float().cpu().numpy()

        print(f"特征形状: {img_features.shape}")
        print(f"特征大小: {img_features.size * 4 / (1024 ** 2):.2f} MB")

        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ",
              img_features.shape)




if __name__ == "__main__":

    # 设置输入和输出路径
    dir_fps_path = '/mnt/sda/shenhao/datasets/MUSIC-AVQA/frames'
    dst_clip_path = '/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/clip_merge'
    # 调用函数提取特征
    ImageClIP_feat_extract(dir_fps_path, dst_clip_path)

# 使用示例
# features = extract_image_features("/mnt/sda/shenhao/datasets/MUSIC-AVQA/frames/00007656/00000030.jpg")
# print(f"特征形状: {features.shape}")
# print(f"特征大小: {features.numel() * 4 / (1024**2):.2f} MB")

# 新增批量处理示例
