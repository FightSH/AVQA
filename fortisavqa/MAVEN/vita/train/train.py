"""
此脚本用于训练 MAVEN 模型。
它处理模型的设置、数据加载、训练参数配置以及训练过程本身。
主要功能包括：
- 设置随机种子以保证可复现性。
- 定义模型参数 (ModelArguments) 和训练参数 (TrainingArguments)。
- 实现与 DeepSpeed ZeRO 相关的辅助函数，用于处理分布式训练中的模型状态。
- 实现 PEFT (Parameter-Efficient Fine-Tuning) 相关的辅助函数，用于获取和保存 LoRA 权重。
- `find_all_linear_names`: 查找模型中所有线性层的名称，用于 LoRA。
- `safe_save_model_for_hf_trainer`: 安全地保存 Hugging Face Trainer 训练的模型。
- `train`: 主训练函数，负责整个训练流程。
"""
import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import set_seed

from .. import conversation as conversation_lib
from ..model import *
from ..train.vita_trainer import VITATrainer

# from vita.util.data_utils_video_audio import make_supervised_data_module, DataArguments
# from vita.util.data_utils_video_audio_neg_patch import make_supervised_data_module, DataArguments
from ..util.data_utils_video_audio_neg_frameCat import DataArguments, make_supervised_data_module


def set_random_seed(seed):
    """
    设置随机种子以确保可复现性。

    Args:
        seed (int): 要设置的随机种子。
    """
    random.seed(seed)  # 为 Python 内置的 random 模块设置随机种子
    np.random.seed(seed)  # 为 numpy 设置随机种子
    torch.manual_seed(seed)  # 为 PyTorch CPU 设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为所有 PyTorch GPU 设置随机种子
    set_seed(seed)  # 为 transformers 库设置随机种子


set_random_seed(42)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    """
    与模型配置相关的参数。
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "预训练模型的路径或 huggingface.co 上的模型标识符。"})
    model_type: Optional[str] = field(default=None, metadata={"help": "模型的类型 (例如 'mixtral-8x7b', 'llama3-8b')。"})
    version: Optional[str] = field(default=None, metadata={"help": "模型的版本，用于选择对话模板。"})
    freeze_backbone: bool = field(default=False, metadata={"help": "是否冻结骨干网络权重。"})
    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "是否调整多模态 MLP 适配器。"})
    vision_tower: Optional[str] = field(default=None, metadata={"help": "视觉塔的路径或名称。"})
    audio_encoder: Optional[str] = field(default=None, metadata={"help": "音频编码器的路径或名称。"})
    freeze_audio_encoder: bool = field(default=True, metadata={"help": "是否冻结音频编码器。"})
    freeze_audio_encoder_adapter: bool = field(default=True, metadata={"help": "是否冻结音频编码器适配器。"})
    unfreeze_vision_tower: bool = field(default=False, metadata={"help": "是否解冻视觉塔。"})
    use_s2: bool = field(default=False, metadata={"help": "是否使用 S2 分词。"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "预训练的多模态 MLP 适配器的路径。"})
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu", metadata={"help": "多模态投影器的类型。"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    与训练过程相关的参数。
    """
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Hugging Face 缓存目录。"})
    optim: str = field(default="adamw_torch", metadata={"help": "使用的优化器。"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "是否移除数据集中未使用的列。"})
    freeze_mm_mlp_adapter: bool = field(default=False, metadata={"help": "是否冻结多模态 MLP 适配器。"})
    mpt_attn_impl: Optional[str] = field(default="triton", metadata={"help": "MPT 模型的注意力实现方式。"})
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = True # metadata={"help": "是否启用 LoRA 微调。"}
    lora_r: int = 64 # metadata={"help": "LoRA 的秩。"}
    lora_alpha: int = 16 # metadata={"help": "LoRA 的 alpha 参数。"}
    lora_dropout: float = 0.05 # metadata={"help": "LoRA 层的 dropout 概率。"}
    lora_weight_path: str = "" # metadata={"help": "LoRA 权重的路径。"}
    lora_bias: str = "none" # metadata={"help": "LoRA 中的偏置类型 ('none', 'all', 'lora_only')。"}
    mm_projector_lr: Optional[float] = None # metadata={"help": "多模态投影器的学习率。如果为 None，则使用全局学习率。"}
    group_by_modality_length: bool = field(default=False, metadata={"help": "是否根据模态长度对样本进行分组，以实现更高效的批处理。"})


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    可能是 DeepSpeed ZeRO Stage 3 的辅助函数，用于获取参数。
    如果参数是 DeepSpeed 管理的参数 (具有 ds_id 属性且状态不是 NOT_AVAILABLE)，
    则使用 GatheredParameters 来获取实际参数值。
    否则，直接返回参数的 detach().cpu().clone()。

    Args:
        param: 模型参数。
        ignore_status (bool, optional): 是否忽略参数的 ds_status。默认为 False。
        name (str, optional): 参数的名称，用于日志记录。默认为 None。

    Returns:
        torch.Tensor: 获取到的参数张量 (在 CPU 上)。
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        # Clear active_sub_modules before gathering parameters
        param.ds_active_sub_modules.clear()
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    # print("1111111111111111111111111111111111111111111111111111")
    return param


# Borrowed from peft.util.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    获取 PEFT (LoRA) 模型的状态字典，可能与 DeepSpeed ZeRO Stage 3 兼容。
    根据 `bias` 参数的不同取值，选择性地包含 LoRA 相关的权重和偏置。

    Args:
        named_params: 模型中所有命名参数的迭代器。
        bias (str): 指定如何处理偏置项。
            - "none": 只包括 LoRA 权重 (名称中包含 "lora_")。
            - "all": 包括 LoRA 权重和所有偏置项 (名称中包含 "bias")。
            - "lora_only": 包括 LoRA 权重和与 LoRA 层相关的偏置。
            - 其他: 抛出 NotImplementedError。

    Returns:
        dict: 包含所选参数的状态字典。
    """
    # print("---------------------bias: ", bias)
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError

    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取 PEFT 模型中非 LoRA 部分的状态字典，可能与 DeepSpeed ZeRO Stage 3 兼容。
    主要用于保存那些在 LoRA 微调之外被更新的参数 (例如，如果某些非 LoRA 部分也被解冻和训练)。

    Args:
        named_params: 模型中所有命名参数的迭代器。
        require_grad_only (bool, optional): 是否只包括需要梯度的参数。默认为 True。

    Returns:
        dict: 包含所选非 LoRA 参数的状态字典 (在 CPU 上)。
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    获取多模态适配器 (mm_projector) 的状态字典，可能与 DeepSpeed ZeRO Stage 3 兼容。
    用于单独保存多模态适配器的权重。

    Args:
        named_params: 模型中所有命名参数的迭代器。
        keys_to_match (list[str]): 用于匹配参数名称的关键字列表 (例如, ["mm_projector"])。

    Returns:
        dict: 包含匹配到的参数的状态字典 (在 CPU 上)。
    """
    to_return = {
        k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """
    查找并返回模型中所有线性层 (torch.nn.Linear) 的名称。
    这通常用于 PEFT (如 LoRA) 中，以确定哪些模块应该被适配器替换或包装。
    会排除多模态相关的关键字 (如 "vision_tower", "vision_resampler") 和 "lm_head"。

    Args:
        model (torch.nn.Module): 要检查的模型。

    Returns:
        list[str]: 模型中所有符合条件的线性层的名称列表。
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    # multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    multimodal_keywords = ["vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # names = name.split(".")
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)  # 使用完整路径

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    # print('--------------------------------------------------------------------\n')
    # print(list(lora_module_names))
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    为 Hugging Face Trainer 安全地保存模型。
    处理了不同的情况：
    - 如果只调整多模态 MLP 适配器 (tune_mm_mlp_adapter=True)，则只保存适配器的权重。
    - 如果使用了 DeepSpeed，则使用 trainer.save_model()。
    - 否则，获取模型的 state_dict，转移到 CPU，然后使用 trainer._save() 保存。

    Args:
        trainer (transformers.Trainer): Hugging Face Trainer 实例。
        output_dir (str): 模型保存的输出目录。
    """

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin")
                )
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank # 设置全局 local_rank，用于 rank0_print

    # 根据训练参数确定计算精度 (float16, bfloat16, 或 float32)
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    assert model_args.vision_tower is not None
    if model_args.model_type in {"mixtral-8x7b"}:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.model_type == "llama3-8b":
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_type == "mixtral-8x7b":
        torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
        model = VITAMixtralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            **bnb_model_from_pretrained_args,
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    else:
        model.model.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 如果启用了 LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        # 配置 LoRA 参数
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model), # 查找所有线性层作为 LoRA 的目标模块
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        # 如果使用16位精度，转换模型类型
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config) # 应用 LoRA 配置到模型

    # 设置对话模板
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]

    model.config.freeze_audio_encoder = model_args.freeze_audio_encoder
    model.config.freeze_audio_encoder_adapter = model_args.freeze_audio_encoder_adapter
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.get_model().initialize_audio_modules(model_args=model_args)

    vision_tower = model.get_vision_tower()
    vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device
    )

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device
    )

    data_args.image_processor = vision_tower.image_processor
    data_args.audio_processor = audio_encoder.audio_processor

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # 根据训练参数配置，决定是否冻结或解冻模型的特定部分
    model.config.tune_mm_mlp_adapter = (
        training_args.tune_mm_mlp_adapter
    ) = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False) # 通常在只微调适配器时，会冻结模型其他部分
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True # 只训练多模态投影器

    if model_args.freeze_audio_encoder:
        for p in model.get_model().audio_encoder.parameters():
            p.requires_grad = False # 冻结音频编码器

    if not model_args.unfreeze_vision_tower:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = False # 冻结视觉塔 (除非明确解冻)

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False # 冻结多模态 MLP 适配器

    if training_args.bits in [4, 8]: # 如果使用4位或8位量化
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device) # 转换多模态投影器的类型

    model.config.mm_projector_lr = training_args.mm_projector_lr # 设置多模态投影器的学习率

    model.config.use_s2 = model_args.use_s2 # 是否使用 S2

    model.config.unfreeze_vision_tower = (
        training_args.unfreeze_vision_tower
    ) = model_args.unfreeze_vision_tower
    if training_args.unfreeze_vision_tower:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True # 解冻视觉塔以进行微调

    # 针对4位或8位量化训练时，对特定模块进行类型转换
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer): # LoRA 层
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name: # 归一化层
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name: # 语言模型头部或词嵌入层
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 创建数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 初始化训练器
    trainer = VITATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # 开始训练，如果存在检查点则从检查点恢复
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state() # 保存训练器状态
    model.config.use_cache = True # 训练完成后，重新启用模型的缓存机制 (如果有)
    # print("traning.local_rank: ", training_args.local_rank)
    # 保存模型权重
    if training_args.lora_enable: # 如果启用了 LoRA
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias) # 获取 LoRA 权重
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters()) # 获取非 LoRA 的可训练权重
        if training_args.local_rank == 0 or training_args.local_rank == -1: # 只在主进程保存
            model.config.save_pretrained(training_args.output_dir) # 保存模型配置
            model.save_pretrained(training_args.output_dir, state_dict=state_dict) # 保存 LoRA 权重
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"), # 保存非 LoRA 的可训练权重
            )
    else: # 如果没有启用 LoRA
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir) # 使用安全保存函数


if __name__ == "__main__":
    train()
