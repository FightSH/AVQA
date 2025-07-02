import torch
import torch.nn as nn
from transformers import SiglipTextModel, SiglipProcessor

from src.models.clip import load


class CLIP_TEncoder(nn.Module):
    def __init__(self,
                 model_name: str
    ):
        super(CLIP_TEncoder, self).__init__()
        
        model = load(model_name, q_aware_N=-1, device='cpu')[0]
        self.dtype = model.dtype
        self.transformer = model.transformer
        self.vocab_size = model.vocab_size
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final

        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        
        del model

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)[:x.shape[1]]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), torch.argmax(text, dim=-1)] @ self.text_projection, x
    
class SigLIP_TEncoder(nn.Module):
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384",model_path: str = '/mnt/sda/shenhao/models'):
        super(SigLIP_TEncoder, self).__init__()
        
        self.model = SiglipTextModel.from_pretrained(model_name,cache_dir=model_path)
        self.processor = SiglipProcessor.from_pretrained(model_name,cache_dir=model_path)
        self.dtype = torch.float32
        
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, text: torch.Tensor):
        """
        Args:
            text: tokenized的tensor，与CLIP_TEncoder保持一致 [batch_size, seq_len]
        Returns:
            tuple: (projected_features, last_hidden_state) 与CLIP_TEncoder输出格式一致
        """
        # 创建attention_mask
        attention_mask = (text != 0).long()
        
        # 获取模型输出
        outputs = self.model(input_ids=text, attention_mask=attention_mask)
        
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 获取EOS token的位置（类似CLIP的argmax操作）
        # 找到每个序列中最后一个非零token的位置
        sequence_lengths = attention_mask.sum(dim=1) - 1  # 减1是因为索引从0开始
        batch_size = text.shape[0]
        
        # 提取EOS token的特征作为句子级表示
        sentence_features = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        
        return sentence_features, last_hidden_state
