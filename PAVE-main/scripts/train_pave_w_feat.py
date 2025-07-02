# Adopted from https://github.com/haotian-liu/LLaVA. The training entrance.
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['WANDB_MODE'] = 'disabled'  # 禁用wandb日志和交互
os.environ['WANDB_DISABLED'] = 'true'  # 兼容性设置，彻底关闭wandb

# 强制设置GPU环境变量
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print(f"CUDA_VISIBLE_DEVICES 设置为: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

import pathlib
import torch
import ipdb
import sys

# 验证GPU设置
print(f"PyTorch可见GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU设备: {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA不可用")

from libs.utils.train_utils import rank0_print

from .multimodal_encoder.builder import build_temporal_aggregator, build_video_tower, build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libs.mm_utils import get_anyres_image_grid_shape, split_list_lengths
from libs.model import *
from libs import conversation_lib as conversation_lib
from libs.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from libs.utils.train_utils import MODEL_ARGUMENTS_MAPPING, DATA_ARGUMENTS_MAPPING




# temporal_aggregator_parameters = [name for name, _ in opt_model.named_parameters() if "temporal_aggregator" in name]
def filter_the_state_dict(state_dict, keyword):
    # filter the state dict using the keyword
    new_state_dict = {key: state_dict[key] for key in state_dict if keyword in key}
    return new_state_dict


def prepare_video_model(training_args, model_args, data_args, compute_dtype, attn_implementation):
    '''
        prepare the model and the tokenizer for training for step1 and step2 training
    '''
    
    # the config of the training bits
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        print(f"使用量化训练，bits={training_args.bits}")
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector", "temporal_aggregator"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    else:
        print("不使用量化训练")
    
    # set a hyper-params for determine whether we use the mplug-owl3
    # ipdb.set_trace() # check the model whether is calling the right class
    use_hyper_attention = model_args.use_hyper_attention if hasattr(model_args, 'use_hyper_attention') else False
    print(f"use_hyper_attention设置为: {use_hyper_attention}")

    # load the LLM
    # It will load LLM + Vision backbone + MLP
    if model_args.video_tower is not None or model_args.vision_tower is not None:
        print("检测到video_tower或vision_tower，加载多模态模型")
        print("video_tower", model_args.video_tower)
        print("vision_tower", model_args.vision_tower)
        if 'qwen2' in model_args.model_name_or_path.lower():
            print("加载PAVEQwen2ForCausalLM模型")
            model = PAVEQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )     
        else:
            print("不支持的模型类型，只支持qwen2")
            raise NotImplementedError
    else:
        print("加载LlamaForCausalLM基础模型")
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    # set the bits used in the training
    if training_args.bits in [4, 8]:
        print("准备量化训练模型")
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    else:
        print("不使用量化训练准备")

    if training_args.gradient_checkpointing:
        print("启用梯度检查点")
        if hasattr(model, "enable_input_require_grads"):
            print("使用enable_input_require_grads方法")
            model.enable_input_require_grads()
        else:
            print("使用register_forward_hook方法")
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    else:
        print("不使用梯度检查点")

    ############ load the tokenizer and reshape the model match the tokenizer
    # load the tokenizer (support Qwen and LLaMA)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # set the tokenizer details for the data preparation
    if model_args.version == "v0":
        print("使用tokenizer版本v0")
        if tokenizer.pad_token is None:
            print("tokenizer.pad_token为None，设置为[PAD]")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
        else:
            print("tokenizer.pad_token已存在")
    elif model_args.version == "v0.5":
        print("使用tokenizer版本v0.5，设置pad_token为unk_token")
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version.startswith("Qwen2Tokenizer"):
        print("使用Qwen2Tokenizer版本")
        tokenizer.pad_token = '<|endoftext|>'
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        print(f"使用其他tokenizer版本: {model_args.version}")
        if tokenizer.pad_token is None:
            print("pad_token为None，设置为unk_token")
            tokenizer.pad_token = tokenizer.unk_token
        else:
            print("pad_token已存在")
        if model_args.version in conversation_lib.conv_templates:
            print(f"使用指定的conversation模板: {model_args.version}")
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            print("使用默认的vicuna_v1 conversation模板")
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # handle the init of the special tokens
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    # handle the special cases that we want to add some tokens to seperate the image and text
    # or we use the start and end special token to seperate the image 
    # usually we do not need this 
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    if use_hyper_attention:
        print("hyper_attention模式：过滤cross-attn模块的LoRA参数")
        selected_module = [ele for ele in selected_module if 'self_attn.v_kv_proj' not in ele and 'self_attn.gate_proj' not in ele]
    else:
        print("普通模式：不过滤cross-attn模块")
    
    print('following layer add lora params: ', selected_module)
    
    # lora setting and init the lora
    if training_args.lora_enable:
        print("启用LoRA训练")
        # target_params = ['v_proj', 'gate_proj', 'up_proj', 'o_proj', 'q_proj', 'down_proj', 'k_proj'] 
        # model.state_dict().keys()
        # 'model.layers.0.self_attn.v_proj.bias'
        # In the module level it has to be model.base_model.layers.named_modules() or model.base_model.layers.state_dict()
        # In the naming it do not need to has 'base_model'
        selected_module = ['model.layers.' + name for name, module in model.base_model.layers.named_modules() if 'proj' in name] # Control the lora only on the LLM
        # ipdb.set_trace() # check the lora module
        if use_hyper_attention:
            print("hyper_attention模式：过滤cross-attn模块的LoRA参数")
            selected_module = [ele for ele in selected_module if 'self_attn.v_kv_proj' not in ele and 'self_attn.gate_proj' not in ele]
        else:
            print("普通模式：不过滤cross-attn模块")
        
        print('following layer add lora params: ', selected_module)
        
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            # target_modules=find_all_linear_names(model),
            target_modules=selected_module,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        
        # unfreeze the cross-attn model params for the training
        if use_hyper_attention:
            print("hyper_attention模式：解冻cross-attn参数")
            for p_n, p in model.named_parameters():
                if 'self_attn.v_kv_proj' in p_n or 'self_attn.gate_proj' in p_n:
                    p.requires_grad = True
        else:
            print("普通模式：不解冻额外参数")
    else:
        print("不使用LoRA训练")

    # ipdb.set_trace() # the loading of the vision
    if 'video-llava' in model_args.model_name_or_path.lower():
        print("检测到video-llava模型，使用特殊处理")
        processor = {'image': None, 'video': None}
        if model.config.mm_image_tower is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=training_args.device, dtype=torch.float16)
            image_processor = image_tower.image_processor
            processor['image'] = image_processor

        if model.config.mm_video_tower is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=training_args.device, dtype=torch.float16)
            video_processor = video_tower.video_processor
            processor['video'] = video_processor
            
        data_args.image_processor = processor['video']
    else:
        print("使用标准视觉处理流程")
        # prepare the vision encoder and may load the pretrained converted
        if model_args.video_tower is not None:
            print("初始化video_tower")
            # copy some data size info into the model config
            # model_args.num_frames = data_args.num_frames
            model_args.image_size = data_args.image_size
            
            # re-instantiate the Video backbone and the SSM
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            # ipdb.set_trace() # check the loading of the model
            # convert the video tower to a specified data type
            video_tower = model.get_video_tower()
            # ipdb.set_trace() # check the loading of the model
            if video_tower is not None:
                video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
                video_tower.data_type = torch.bfloat16 if training_args.bf16 else torch.float16
            # video_tower.to(dtype=compute_dtype, device=training_args.device) # control the type of the backbone

            # prepare the dataset config
            model.config.tokenizer_padding_side = tokenizer.padding_side
            model.config.tokenizer_model_max_length = tokenizer.model_max_length

            # determin whether we train the temporal_aggregator 
            model.config.tune_temporal_aggregator = training_args.tune_temporal_aggregator = model_args.tune_temporal_aggregator
            if model_args.tune_temporal_aggregator:
                print("启用temporal_aggregator训练")
                model.requires_grad_(False)
                for p in model.get_model().temporal_aggregator.parameters():  # 🔥 解冻参数
                    p.requires_grad = True

            # 量化设置
            if training_args.bits in [4, 8]:
                print("量化模式：设置temporal_aggregator数据类型")
                model.get_model().temporal_aggregator.to(dtype=compute_dtype, device=training_args.device)  # 🔥 设置数据类型
        else:
            print("没有video_tower，跳过视频相关初始化")
    
         # handle the image processor 
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            data_args.image_processor = vision_tower.image_processor
    # ipdb.set_trace() # some further check of the module frozen and the module training  

    # handle the bit in training
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    return model, tokenizer


def load_trained_model_for_eval(model_path, model_base, model_name, 
                                model_arg_name='default',
                                data_arg_name='default',
                                load_8bit=False, load_4bit=False, 
                                device_map="auto", device="cuda", 
                                use_flash_attn=False, **kwargs):
    # reduce the warning output
    warnings.filterwarnings("ignore")
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'pave' in model_name.lower():
        print("检测到PAVE模型")
        if 'lora' in model_name.lower() and model_base is None:
            logger.warning("模型名称包含lora但未提供model_base")
            warnings.warn('There is `lora` in model name but no `model_base` is provided. \
                          If you are loading a LoRA model, please provide the `model_base` argument. \
                          Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        
        if 'lora' in model_name.lower() and model_base is not None:
            print("加载LoRA模型")
            from libs.model.language_model.pave_qwen2 import PAVEQwen2Config, PAVEQwen2ForCausalLM

            base_model_cfg = PAVEQwen2Config.from_pretrained(model_base)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = PAVEQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=base_model_cfg, **kwargs)
            lora_cfg_pretrained = PAVEQwen2Config.from_pretrained(model_path)
            
            # reshaping the language head of the model
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                print('re-initing the lm_head')
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            ## init the vision module 
            print('Init the vision module ...')
            # merge the training config with the lora config
            # the lora_cfg_pretrained contains the parameters sended in through command line
            # the default_model_arg contains the default model parameters
            default_model_arg = MODEL_ARGUMENTS_MAPPING[model_arg_name]
            default_data_args = DATA_ARGUMENTS_MAPPING[data_arg_name]
            print('Warning: we are using MODEL_ARGUMENTS_MAPPING:', model_arg_name, 'DATA_ARGUMENTS_MAPPING:', data_arg_name)
            
            # set the value in lora_cfg_pretrained as default_model_arg, we should use lora_cfg_pretrained latter on
            for key in default_model_arg.__dict__:
                if not key.startswith('__'):
                    if not hasattr(lora_cfg_pretrained, key):
                        setattr(lora_cfg_pretrained, key, default_model_arg.__dict__[key])
            
            # for key in lora_cfg_pretrained.__dict__:
            #     if not key.startswith('__'):
            #         print(key)
            
            # re-instantiate the Video backbone and the SSM
            lora_cfg_pretrained.image_size = default_data_args.image_size
            model.get_model().initialize_vision_modules(
                model_args=lora_cfg_pretrained,
                fsdp=None,
            ) 
            
            # load the pretrained temporal aggregator weights
            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                print(f"从本地加载权重: {model_path}")
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                print("从HuggingFace Hub加载权重")
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(filter_the_state_dict(non_lora_trainables, 'temporal_aggregator'), strict=False)
            
            # handle the special case for the mplug-owl3
            if 'mplug' in model_name.lower():
                print("检测到mplug模型，加载额外参数")
                additional_params_1 = filter_the_state_dict(non_lora_trainables, 'self_attn.v_kv_proj')
                model.load_state_dict(additional_params_1, strict=False)
                additional_params_2 = filter_the_state_dict(non_lora_trainables, 'self_attn.gate_proj')
                model.load_state_dict(additional_params_2, strict=False)                
            else:
                print("非mplug模型，跳过额外参数加载")

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            # import ipdb
            # ipdb.set_trace() # check before merge
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            
            ### handle the loading of the tokenizer
            # lora_cfg_pretrained.pretrain_temporal_aggregator = os.path.join(model_path, 'non_lora_trainables.bin')
            
            model.initialize_vision_tokenizer(lora_cfg_pretrained, tokenizer=tokenizer)
            # ipdb.set_trace() # check the loading of the tokenizer, the size of the tokenizer
            
        elif 'adaptor' in model_name.lower() and model_base is not None: # for the case we only train the adaptor
            print("加载adaptor模型")
            from libs.model.language_model.pave_qwen2 import PAVEQwen2Config

            # init the base LLM model
            base_model_cfg = PAVEQwen2Config.from_pretrained(model_base)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = PAVEQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=base_model_cfg, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                print('re-initing the lm_head')
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            ## init the vision module 
            print('Init the vision module ...')
            cfg_pretrained = PAVEQwen2Config.from_pretrained(model_path)
            
            # merge the training config with the lora config
            # the cfg_pretrained contains the parameters sended in through command line
            # the default_model_arg contains the default model parameters
            default_model_arg = MODEL_ARGUMENTS_MAPPING[model_arg_name]
            default_data_args = DATA_ARGUMENTS_MAPPING[data_arg_name]
            print('Warning: we are using MODEL_ARGUMENTS_MAPPING:', model_arg_name, 'DATA_ARGUMENTS_MAPPING:', data_arg_name)
            
            # set the value in cfg_pretrained as default_model_arg, we should use cfg_pretrained latter on
            for key in default_model_arg.__dict__:
                if not key.startswith('__'):
                    if not hasattr(cfg_pretrained, key):
                        setattr(cfg_pretrained, key, default_model_arg.__dict__[key])
            
            # for key in lora_cfg_pretrained.__dict__:
            #     if not key.startswith('__'):
            #         print(key)
            
            # re-instantiate the Video backbone and the SSM
            cfg_pretrained.image_size = default_data_args.image_size
            model.get_model().initialize_vision_modules(
                model_args=cfg_pretrained,
                fsdp=None,
            ) 
            
            # load the pretrained temporal aggregator weights
            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                print(f"从本地加载adaptor权重: {model_path}")
                non_lora_trainables = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            else:
                print("从HuggingFace Hub加载adaptor权重")
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'mm_projector.bin')
                
            # ipdb.set_trace() # check the loaded weight names
            # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            # if any(k.startswith('model.model.') for k in non_lora_trainables):
            #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(filter_the_state_dict(non_lora_trainables, 'temporal_aggregator'), strict=False)
            
            # resize and handle the size of the head
            model.initialize_vision_tokenizer(cfg_pretrained, tokenizer=tokenizer)
            # ipdb.set_trace() # check the loaded LLM and temporal aggregator
        else:
            logger.error("不支持的PAVE模型配置")
            raise NotImplementedError
    else:
        logger.error("不支持的模型类型")
        raise NotImplementedError
    
    image_processor = None
    # ipdb.set_trace()
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    # ipdb.set_trace()
    return tokenizer, model, image_processor, context_len