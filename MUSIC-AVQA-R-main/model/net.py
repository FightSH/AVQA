import torch
import torch.nn as nn
from transformers import VisualBertModel
from model.layer import MLP
from utils import grad_mul_const


class RAVQA(nn.Module):
    '''
    RAVQA类：音视频问答(Audio-Visual Question Answering)模型
    该模型使用VisualBert架构来融合视觉、音频和文本特征，进行多模态推理
    '''
    def __init__(self, config):
        '''
        初始化函数
        参数:
            config: 包含模型配置信息的字典
        '''
        super(RAVQA, self).__init__()
        # 使用预训练的VisualBert模型，该模型在COCO数据集上进行过VQA任务的预训练
        self.config = config
        # self.visual_bert = VisualBertModel.from_pretrained("./vlbert")
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        
        # 特征映射层：将不同模态的特征映射到适合VisualBert的维度
        self.audio_linear = nn.Linear(128, 768)  # 音频特征从128维映射到768维
        self.video_linear = nn.Linear(512, 768)  # 视频特征从512维映射到768维
        self.av_linear = nn.Linear(768, 2048)    # 将音频和视频特征进一步映射到2048维(VisualBert视觉输入维度)
        
        self.dropout = nn.Dropout(p=0.1)  # 防止过拟合的Dropout层
        self.cls = nn.Linear(768, config['hyper_para']['num_labels'])  # 分类器，输出类别数量由配置决定

        # 偏置学习器(bias learner)：用于处理不同模态的偏置
        # 问题偏置学习器
        if config['hyper_para']['bias_learner']['q_bias']:
            self.q_bias = MLP(dimensions=config['hyper_para']['mlp']['dimensions'])
        # 音频偏置学习器
        if config['hyper_para']['bias_learner']['a_bias']:
            self.a_bias = MLP(dimensions=config['hyper_para']['mlp']['dimensions'])
        # 视频偏置学习器
        if config['hyper_para']['bias_learner']['v_bias']:
            self.v_bias = MLP(dimensions=config['hyper_para']['mlp']['dimensions'])

    def forward(self, b_inputs):
        '''
        前向传播函数
        参数:
            b_inputs: 包含各种输入数据的字典
        返回:
            一个包含各种输出logits的字典
        '''
        inputs, position_ids = b_inputs['inputs'], b_inputs['pos'].cuda()  # 文本输入和位置ID
        audio_emb, video_emb = b_inputs['audio_emb'].cuda(), b_inputs['video_emb'].cuda()  # 音频和视频嵌入
        audio_token_type_ids, video_token_type_ids = b_inputs['audio_token_type'].cuda(), b_inputs['video_token_type'].cuda()  # 音频和视频的token类型ID

        # 处理音频特征
        audio_emb_mid = self.audio_linear(audio_emb)  # 音频特征初步映射(128->768)
        audio_emb = self.av_linear(audio_emb_mid)    # 音频特征进一步映射(768->2048)
        
        # 处理视频特征
        video_emb_mid = self.video_linear(video_emb)  # 视频特征初步映射(512->768)
        video_emb = self.av_linear(video_emb_mid)    # 视频特征进一步映射(768->2048)
        
        # 合并音频和视频特征，作为视觉输入
        av_feat = torch.cat([video_emb, audio_emb], dim=1)
        av_token_type_ids = torch.cat([video_token_type_ids, audio_token_type_ids], dim=1)

        # 将所有输入数据移至GPU
        inputs = {key: inputs[key].cuda() for key in inputs}
        inputs.update({
            'visual_embeds': av_feat,  # 音视频融合特征作为视觉输入
            'visual_token_type_ids': av_token_type_ids,  # 对应的token类型ID
            'position_ids': position_ids  # 位置编码
        })
        
        # 使用VisualBert进行多模态融合
        outputs = self.visual_bert(**inputs)
        pooler_output = outputs.pooler_output  # 获取[CLS]令牌的输出，代表整个序列的表示
        fusion_logits = self.cls(self.dropout(pooler_output))  # 通过分类器得到融合后的logits

        # 根据配置计算不同模态的偏置logits
        q_bias_logits, a_bias_logits, v_bias_logits = None, None, None
        if self.config['hyper_para']['bias_learner']['q_bias']:
            q_bias_logits = self.get_bias_classifier_logits_q(inputs)  # 问题偏置
        if self.config['hyper_para']['bias_learner']['a_bias']:
            a_bias_logits = self.get_bias_classifier_logits_a(audio_emb_mid)  # 音频偏置
        if self.config['hyper_para']['bias_learner']['v_bias']:
            v_bias_logits = self.get_bias_classifier_logits_v(video_emb_mid)  # 视频偏置

        # 返回所有计算得到的logits
        return {
            'fusion_logits': fusion_logits,  # 融合后的主要logits
            'q_bias_logits': q_bias_logits,  # 问题偏置logits
            'a_bias_logits': a_bias_logits,  # 音频偏置logits
            'v_bias_logits': v_bias_logits   # 视频偏置logits
        }

    def get_bias_classifier_logits_q(self, inputs):
        '''
        获取问题偏置分类器的logits
        参数:
            inputs: VisualBert的文本输入
        返回:
            问题偏置的logits
        '''
        que_emb = self.visual_bert(
            input_ids=inputs['input_ids'],
            position_ids=inputs['position_ids'],
            attention_mask=inputs['attention_mask']
        )
        #que_emb = grad_mul_const(que_emb.pooler_output, 0.0)  # 梯度乘以常数(已注释)
        q_bias_logits = self.q_bias(input)  # 通过MLP获取问题偏置logits
        return q_bias_logits

    def get_bias_classifier_logits_a(self, inputs):
        audio_emb = self.visual_bert(inputs_embeds=inputs)
        #audio_emb = grad_mul_const(audio_emb.pooler_output, 0.0)
        a_bias_logits = self.a_bias(audio_emb.pooler_output)
        return a_bias_logits

    def get_bias_classifier_logits_v(self, inputs):
        '''
        获取视频偏置分类器的logits
        参数:
            inputs: 视频嵌入表示
        返回:
            视频偏置的logits
        '''
        video_emb = self.visual_bert(inputs_embeds=inputs)  # 将视频嵌入输入到VisualBert
        #video_emb = grad_mul_const(video_emb.pooler_output, 0.0)  # 梯度乘以常数(已注释)
        v_bias_logits = self.v_bias(input)  # 通过MLP获取视频偏置logits
        return v_bias_logits
