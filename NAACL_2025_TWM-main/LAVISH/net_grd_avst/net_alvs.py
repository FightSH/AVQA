import torch

import torch.nn as nn
from einops import rearrange, repeat
from net_encoder import AVQA_Fusion_Net, AMS




class AVQA_Net(nn.Module):

    def __init__(self, opt, mode="visual"):
        super(AVQA_Net, self).__init__()

        self.opt = opt
        self.mode = mode
        self.visual_encoder = AVQA_Fusion_Net(opt)

        # ===================================== load pretrained model ===============================================
        ####### concat model
        pretrained_file = "/mnt/sda/shenhao/code/NAACL_2025_TWM/LAVISH/grounding_gen/main_grounding_gen_best.pt"
        checkpoint = torch.load(pretrained_file)
        print("\n-------------- loading pretrained models --------------")
        model_dict = self.visual_encoder.state_dict()
        tmp = ['fc_a1.weight', 'fc_a1.bias', 'fc_a2.weight', 'fc_a2.bias',
               'fc_gl.weight', 'fc_gl.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight',
               'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias']
        tmp2 = ['fc_a1.weight', 'fc_a1.bias', 'fc_a2.weight', 'fc_a2.bias']
        pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        pretrained_dict2 = {str(k).split('.')[0] + '.' + str(k).split('.')[1] + '_pure.' + str(k).split('.')[-1]: v for
                            k, v in checkpoint.items() if k in tmp2}

        model_dict.update(pretrained_dict1)  # 利用预训练模型的参数，更新模型
        model_dict.update(pretrained_dict2)  # 利用预训练模型的参数，更新模型
        self.visual_encoder.load_state_dict(model_dict)

        print("\n-------------- load pretrained models --------------")

        # ===================================== load pretrained model ===============================================

        self.visual_encoder.frozen_model()
        self.audio_encoder = AMS(1024, 512, 8, "cuda")
        self.visual_mapping = nn.Linear(512, 512)

    def frozen_model(self):
        for p in self.parameters():
            p.requires_grad = False
        return


    def forward(self, audio, visual_posi, question):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
        '''
        print(f"audio shape: ", {audio.shape})
        print(f"visual_posi shape: ", {visual_posi.shape})
        print(f"question shape: ", {question.shape})
        bs, t, c, h, w = visual_posi.shape

        audio_re = audio.view(audio.size(0) * audio.size(1), -1)
        print(f"audio_re shape: {audio_re.shape}")
        bs = visual_posi.size(0)
        f_v, _ = self.visual_encoder(audio, visual_posi)
        print(f"f_v shape: {f_v.shape}")
        f_v = f_v.squeeze()
        f_v = self.visual_mapping(f_v)
        if self.mode == "visual":
            f_qst_ = self.visual_encoder.qst_feature_forward(question)
            rep_f_qst = repeat(f_qst_, 'b d -> b t d', t=t)
            f_qst = rearrange(rep_f_qst, 'b t d -> (b t) d')
            f_a = None
        else:
            f_a, _ = self.audio_encoder(audio_re, f_v)
            f_qst = None

        return f_v, f_a, f_qst


    def forward_visual(self, audio, visual_posi, question, mixup_lambda, stage='eval'):
        audio, visual_posi, question = audio.unsqueeze(0), visual_posi.unsqueeze(0), question.unsqueeze(0)
        bs, t, c, h, w = visual_posi.shape

        audio_re = audio.view(audio.size(0) * audio.size(1), -1)
        bs = visual_posi.size(0)
        f_v, _ = self.visual_encoder(audio, visual_posi)
        f_v = f_v.squeeze()
        f_v = self.visual_mapping(f_v)
        if self.mode == "visual":
            f_qst_ = self.visual_encoder.qst_feature_forward(question)
            rep_f_qst = repeat(f_qst_, 'b d -> b t d', t=t)
            f_qst = rearrange(rep_f_qst, 'b t d -> (b t) d')
            f_a = None
        else:
            f_a, _ = self.audio_encoder(audio_re, f_v)
            f_qst = None
        return f_v, f_a, f_qst
