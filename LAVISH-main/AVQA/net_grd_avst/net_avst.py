import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from einops import rearrange, repeat

class VisualAdapter(nn.Module):
	"""
	VisualAdapter 视觉适配器模块，用于在主干网络中插入可训练的适配器层，实现特征的高效调整和多模态交互。
	支持三种类型：bottleneck（可多模态）、basic。
	"""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt

		# 是否使用门控机制
		if use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		# bottleneck结构，支持多模态交互
		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // reduction_factor

			# 多模态交互的token参数
			self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
			self.gate_av = nn.Parameter(torch.zeros(1))

			# 激活函数
			self.activation = nn.ReLU(inplace=True)
			# 下采样卷积
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			# 上采样卷积
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			# 可选的BN层
			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)
			
			# 可选的LayerNorm
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)

		# 普通bottleneck结构
		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // reduction_factor
			self.activation = nn.ReLU(inplace=True)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)
			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)

		# basic结构，使用线性层
		elif adapter_kind == "basic":
			self.activation = nn.ReLU(inplace=True)
			self.conv = nn.Linear(input_dim, output_dim, bias=False)
			if use_bn:
				self.bn = nn.BatchNorm1d(output_dim)
		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		"""
		前向传播，输入特征x和可选的视觉token，实现特征的适配和多模态交互。
		参数:
			x: 输入特征，形状[N, C, L, 1]或[N, C, L]
			vis_token: 可选的视觉token，用于多模态交互
		返回:
			output: 适配后的特征
		"""
		# 多模态bottleneck结构
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			# 生成重复的token用于交互
			# `repeat(self.my_tokens, 't d -> b t d', b=x.size(0))` 这行代码的作用是：
				# - `self.my_tokens` 是形状为 `[num_tokens, input_dim]` 的参数张量。
				# - 通过 `einops.repeat`，将 `self.my_tokens` 在 batch 维度上复制 `b` 次（`b = x.size(0)`，即 batch size），得到形状为 `[b, num_tokens, input_dim]` 的张量。
				# - 这样每个 batch 都有一份相同的 token 参数，用于后续与输入特征进行多模态交互。
			# 简而言之：**把 token 参数扩展到 batch 维度，每个样本一份，便于后续批量计算。**
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
			# 计算视觉token与自有token的注意力
			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))
			att_v2tk = F.softmax(att_v2tk, dim=-1)
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))
			rep_token = rep_token + rep_token_res
			# token与输入特征交互
			att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))
			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)
			# 门控融合交互特征
			x = x + self.gate_av*x_res.contiguous()
			# 可选的LayerNorm
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
			# 下采样卷积
			z = self.down_sampler(x)
			if self.use_bn:
				z = self.bn1(z)
			z = self.activation(z)
			# 上采样卷积
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
	
		# 普通bottleneck结构
		elif self.adapter_kind == "bottleneck":
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
			z = self.down_sampler(x)
			if self.use_bn:
				z = self.bn1(z)
			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
			
		# basic结构
		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')

		# 可选的输出后LayerNorm
		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
		# 可选的门控
		if self.gate is not None:
			output = self.gate * output

		return output



def batch_organize(out_match_posi, out_match_nega):
	# audio B 512
	# posi B 512
	# nega B 512

	out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
	batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
	for i in range(out_match_posi.shape[0]):
		out_match[i * 2, :] = out_match_posi[i, :]
		out_match[i * 2 + 1, :] = out_match_nega[i, :]
		batch_labels[i * 2] = 1
		batch_labels[i * 2 + 1] = 0
	
	return out_match, batch_labels

# Question
class QstEncoder(nn.Module):

	def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

		super(QstEncoder, self).__init__()
		self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
		self.tanh = nn.Tanh()
		self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
		self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

	def forward(self, question):

		qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
		qst_vec = self.tanh(qst_vec)
		qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
		self.lstm.flatten_parameters()
		_, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
		qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
		qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
		qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
		qst_feature = self.tanh(qst_feature)
		qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

		return qst_feature


class AVQA_Fusion_Net(nn.Module):
	"""
	AVQA_Fusion_Net 是多模态音视频问答任务的主网络结构。
	该网络融合了音频、正样本视频、负样本视频和问题文本特征，进行音视频匹配和问答预测。
	"""

	def __init__(self, opt):
		super(AVQA_Fusion_Net, self).__init__()

		self.opt = opt

		# 音频特征处理的全连接层
		self.fc_a1 =  nn.Linear(128, 512)
		self.fc_a2=nn.Linear(512,512)

		self.fc_a1_pure =  nn.Linear(128, 512)
		self.fc_a2_pure=nn.Linear(512,512)
		# self.visual_net = resnet18(pretrained=True)

		# 视频特征处理的全连接层
		self.fc_v = nn.Linear(2048, 512)
		# self.fc_v = nn.Linear(1536, 512)
		self.fc_st = nn.Linear(512, 512)
		self.fc_fusion = nn.Linear(1024, 512)
		self.fc = nn.Linear(1024, 512)
		self.fc_aq = nn.Linear(512, 512)
		self.fc_vq = nn.Linear(512, 512)

		# 多层感知机用于特征融合
		self.linear11 = nn.Linear(512, 512)
		self.dropout1 = nn.Dropout(0.1)
		self.linear12 = nn.Linear(512, 512)

		self.linear21 = nn.Linear(512, 512)
		self.dropout2 = nn.Dropout(0.1)
		self.linear22 = nn.Linear(512, 512)
		self.norm1 = nn.LayerNorm(512)
		self.norm2 = nn.LayerNorm(512)
		self.dropout3 = nn.Dropout(0.1)
		self.dropout4 = nn.Dropout(0.1)
		self.norm3 = nn.LayerNorm(512)

		# 多头注意力机制，分别用于音频和视频
		self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
		self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)

		# 问题编码器
		self.question_encoder = QstEncoder(93, 512, 512, 1, 512)

		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(0.5)
		self.fc_ans = nn.Linear(512, 42)  # 答案分类器

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc_gl=nn.Linear(1024,512)

		# 融合音视频特征的多层感知机
		self.fc1 = nn.Linear(1024, 512)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(512, 256)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(256, 128)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(128, 2)
		self.relu4 = nn.ReLU()

		self.yb_fc_v = nn.Linear(1536, 512)
		self.yb_fc_a = nn.Linear(1536, 512)

		# 使用Swin Transformer作为视觉主干网络
		self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)

		### ------------> for swin 
		# 统计Swin各层的隐藏维度信息，用于适配器构建
		hidden_list = []
		down_in_dim = []
		down_out_dim = []
		for idx_layer, my_blk in enumerate(self.swin.layers) :
			if not isinstance(my_blk.downsample, nn.Identity):
				down_in_dim.append(my_blk.downsample.reduction.in_features)
				down_out_dim.append(my_blk.downsample.reduction.out_features)

			for blk in my_blk.blocks:
				hidden_d_size = blk.norm1.normalized_shape[0]
				hidden_list.append(hidden_d_size)
		### <--------------

		# 第一阶段音频适配器
		self.audio_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		# 第一阶段视觉适配器
		self.vis_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

		# 第二阶段音频适配器
		self.audio_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		# 第二阶段视觉适配器
		self.vis_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

	def forward(self, audio, visual_posi, visual_nega, question, stage='eval'):
		'''
		前向传播函数，输入音频、正负视频样本和问题，输出问答预测和音视频匹配分数。

		参数:
			audio:        [B, T, C]，音频特征
			visual_posi:  [B, T, C, H, W]，正样本视频特征
			visual_nega:  [B, T, C, H, W]，负样本视频特征
			question:     [B, T]，问题文本
			stage:        阶段标志，默认'eval'
		返回:
			out_qa:         问答预测结果
			out_match_posi: 正样本匹配分数
			out_match_nega: 负样本匹配分数
		'''
		
		# 获取输入的batch和时空维度信息
		bs,t,c,h,w = visual_posi.shape

		# 音频特征扩展为3通道，适配视觉主干网络
		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
		audio = rearrange(audio, 'b t c w h -> (b t) c w h')

		###### ---------------->
		# 提取音频patch特征
		f_a = self.swin.patch_embed(audio)

		# 提取正样本视频patch特征
		visual_posi = rearrange(visual_posi, 'b t c w h -> (b t) c w h')
		f_v = self.swin.patch_embed(visual_posi)
		# f_v_neg = self.swin.patch_embed(visual_nega)
		idx_layer = 0
		multi_scale = []
		
		idx_block = 0

		# 逐层遍历Swin主干网络，进行音视频特征交互和适配
		for _, my_blk in enumerate(self.swin.layers) :

			for blk in my_blk.blocks:
				# 第一阶段适配器，音视频交互
				# permute(0,2,1) 是将张量的第 2 和第 3 个维度（通常是 [batch, seq\_len, dim]）交换为 [batch, dim, seq\_len]。
				# unsqueeze(-1) 在最后增加一个新维度，变成 [batch, dim, seq\_len, 1]。
				# 这样做的目的是将特征维度 dim 放到第 2 维，方便后续的 2D 卷积等操作。
				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				# Swin主干自注意力与适配器输出融合
				f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.norm1(blk._attn(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)

				# 第二阶段适配器，音视频交互
				f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				# Swin主干MLP与适配器输出融合
				f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path2(blk.norm2(blk.mlp(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)

				idx_layer = idx_layer +1
			#####
			# 下采样
			f_v = my_blk.downsample(f_v)
			f_a = my_blk.downsample(f_a)

		# Swin主干最终归一化
		f_v = self.swin.norm(f_v)
		f_a = self.swin.norm(f_a)

		# 线性映射到512维
		f_v = self.yb_fc_v(f_v)
		f_a = self.yb_fc_a(f_a)
			
		# 负样本视频特征提取（不参与梯度计算）
		with torch.no_grad():
			visual_nega = rearrange(visual_nega, 'b t c h w -> (b t) c h w')
			visual_nega = self.swin.forward_features(visual_nega)

		############## <----------

		# 负样本视频特征映射
		visual_nega = self.yb_fc_v(visual_nega)
		# 正负样本视频特征reshape为[B, T, C, H, W]
		visual_posi = rearrange(f_v, '(b t) (h w) c -> b t c h w', b=bs ,t=t, h=6 ,w=6)
		visual_nega = rearrange(visual_nega, '(b t) (h w) c -> b t c h w', b=bs ,t=t, h=6 ,w=6)

		### -------> yb: cvpr use
		# 音频特征时序平均
		f_a = f_a.mean(dim=1)
		audio = rearrange(f_a, '(b t) c -> b t c', b=bs ,t=t)
		### <-----

		## 问题特征提取
		qst_feature = self.question_encoder(question)
		xq = qst_feature.unsqueeze(0)

		## 音频特征处理
		# audio_feat = F.relu(self.fc_a1(audio))
		audio_feat = F.relu(audio) # yb: for swin only
		audio_feat = self.fc_a2(audio_feat)  
		audio_feat_pure = audio_feat
		B, T, C = audio_feat.size()             # [B, T, C]
		audio_feat = audio_feat.view(B*T, C)    # [B*T, C]

		## 正样本视频特征处理
		B, T, C, H, W = visual_posi.size()
		temp_visual = visual_posi.view(B*T, C, H, W)            # [B*T, C, H, W]
		v_feat = self.avgpool(temp_visual)                      # [B*T, C, 1, 1]
		visual_feat_before_grounding_posi = v_feat.squeeze()    # [B*T, C]

		(B, C, H, W) = temp_visual.size()
		v_feat = temp_visual.view(B, C, H * W)                      # [B*T, C, HxW]
		v_feat = v_feat.permute(0, 2, 1)                            # [B, HxW, C]
		visual_feat_posi = nn.functional.normalize(v_feat, dim=2)   # [B, HxW, C]

		## 音视频定位（grounding）正样本
		audio_feat_aa = audio_feat.unsqueeze(-1)                        # [B*T, C, 1]
		audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B*T, C, 1]
	   
		x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze() # [B*T, HxW]

		x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
		visual_feat_grd = torch.matmul(x2_p, visual_feat_posi)
		visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [B*T, C]   

		# 拼接定位前后的视频特征
		visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi),dim=-1)
		visual_feat_grd = self.tanh(visual_gl)
		visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]

		# 拼接音频和定位后的视频特征
		feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 1024]

		# 多层感知机进行特征融合和匹配分数预测
		feat = F.relu(self.fc1(feat))       # (1024, 512)
		feat = F.relu(self.fc2(feat))       # (512, 256)
		feat = F.relu(self.fc3(feat))       # (256, 128)
		out_match_posi = self.fc4(feat)     # (128, 2)

		###############################################################################################
		# 负样本视频特征处理，流程同正样本
		B, T, C, H, W = visual_nega.size()
		temp_visual = visual_nega.view(B*T, C, H, W)
		v_feat = self.avgpool(temp_visual)
		visual_feat_before_grounding_nega = v_feat.squeeze() # [B*T, C]

		(B, C, H, W) = temp_visual.size()
		v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
		v_feat = v_feat.permute(0, 2, 1)        # [B, HxW, C]
		visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

		##### av grounding nega
		x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
		x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
		visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
		visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()    # [B*T, C]   

		visual_gl=torch.cat((visual_feat_before_grounding_nega,visual_feat_grd_after_grounding_nega),dim=-1)
		visual_feat_grd=self.tanh(visual_gl)
		visual_feat_grd_nega=self.fc_gl(visual_feat_grd)    # [B*T, C]

		# 拼接音频和负样本定位后的视频特征
		feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)   # [B*T, C*2], [B*T, 1024]

		feat = F.relu(self.fc1(feat))       # (1024, 512)
		feat = F.relu(self.fc2(feat))       # (512, 256)
		feat = F.relu(self.fc3(feat))       # (256, 128)
		out_match_nega = self.fc4(feat)     # (128, 2)

		###############################################################################################

		# B = batch size
		B = xq.shape[1]
		# 正样本定位后的视频特征reshape为[T, B, 512]
		visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 512)   # [B, T, 512]
		visual_feat_grd=visual_feat_grd_be.permute(1,0,2)
		
		## 问题作为query，对视频特征做注意力
		visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
		src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
		visual_feat_att = visual_feat_att + self.dropout2(src)
		visual_feat_att = self.norm1(visual_feat_att)
	
		# 问题作为query，对音频特征做注意力
		audio_feat_be=audio_feat_pure.view(B, -1, 512)
		audio_feat = audio_feat_be.permute(1, 0, 2)
		audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
		src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
		audio_feat_att = audio_feat_att + self.dropout4(src)
		audio_feat_att = self.norm2(audio_feat_att)
		
		# 融合音频和视频的注意力特征与均值特征
		feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(), visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
		feat = self.tanh(feat)
		feat = self.fc_fusion(feat)

		## 与问题特征融合
		combined_feature = torch.mul(feat, qst_feature)
		combined_feature = self.tanh(combined_feature)
		out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]

		return out_qa, out_match_posi,out_match_nega