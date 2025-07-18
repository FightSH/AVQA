from __future__ import print_function
import sys 
# sys.path.append("/home/guangyao_li/projects/avqa/music_avqa_camera_ready") 
import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo 
import os
from datetime import datetime
import time
args = BaseOptions().parse()

mygpu = GPUInfo.get_info()[0]
gpu_source = {}


import logging

# 获取时间戳字符串（不带/，便于文件名拼接）
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(args.model_save_dir, exist_ok=True)

log_path = os.path.join(args.model_save_dir, f"train_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a'),
        logging.StreamHandler()
    ]
)


# if 'N/A' in mygpu.keys():
# 	for info in mygpu['N/A']:
# 		if info in gpu_source.keys():
# 			gpu_source[info] +=1
# 		else:
# 			gpu_source[info] =1

# for gpu_id in args.gpu:
# 	gpu_id = str(gpu_id)

# 	if gpu_id not in gpu_source.keys():
# 		logging.info(f'go gpu: {gpu_id}')
# 		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# 		break
# 	elif gpu_source[gpu_id] < 1:
# 		logging.info(f'go gpu: {gpu_id}')
# 		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# 		break

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # 导入自动混合精度相关模块
from ipdb import set_trace
from dataloader_avst import *
# from dataloader_avst_bk import *
from net_avst import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb
# from .net_avst import AVQA_Fusion_Net

import warnings
from datetime import datetime
import wandb
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/net_avst/'+TIMESTAMP)

import certifi
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())

logging.info("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")

def batch_organize(out_match_posi,out_match_nega):

	# audio B 512
	# posi B 512
	# nega B 512

	# print("audio data: ", audio_data.shape)
	out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
	batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
	for i in range(out_match_posi.shape[0]):
		out_match[i * 2, :] = out_match_posi[i, :]
		out_match[i * 2 + 1, :] = out_match_nega[i, :]
		batch_labels[i * 2] = 1
		batch_labels[i * 2 + 1] = 0
	
	return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion, epoch):
	model.train()
	total_qa = 0
	correct_qa = 0
	# 定义梯度累积步数
	accumulation_steps = 4  # 可以根据需要调整此值
	
	# 跟踪累积损失，用于日志记录
	accumulated_loss = 0.0
	
	# 创建 GradScaler 实例用于混合精度训练
	scaler = GradScaler()
	
	optimizer.zero_grad()  # 在开始时清零梯度而不是每个批次
	
	for batch_idx, sample in enumerate(train_loader):
		audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

		# 使用 autocast 上下文管理器启用混合精度
		with autocast():
			out_qa, out_match_posi, out_match_nega = model(audio, visual_posi, visual_nega, question, 'train')
			out_match,match_label=batch_organize(out_match_posi,out_match_nega)  
			out_match,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()

			loss_match=criterion(out_match,match_label)
			loss_qa = criterion(out_qa, target)
			loss = loss_qa + 0.5*loss_match
			
			# 如果batch大小不是平均分配的，需要根据实际批次大小缩放损失
			loss = loss / accumulation_steps
		
		# 统计反向传播耗时
		torch.cuda.synchronize()
		start_time = time.time()
		
		# 使用 scaler 进行反向传播
		scaler.scale(loss).backward()
		
		torch.cuda.synchronize()
		backward_time = time.time() - start_time
		
		# 累积用于日志显示的损失
		accumulated_loss += loss.item()

		# 只有在累积了指定步数的梯度后才更新参数
		if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
			# 使用 scaler 更新参数
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()
			
			# 记录训练日志（基于累积的步数）
			if batch_idx % args.log_interval == 0:
				current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				logging.info('[{time}] Train Epoch: {epoch} [{processed}/{total} ({percent:.0f}%)]\tLoss: {loss:.6f}'.format(
					time=current_time,
					epoch=epoch,
					processed=batch_idx * len(audio),
					total=len(train_loader.dataset),
					percent=100. * batch_idx / len(train_loader),
					loss=accumulated_loss * accumulation_steps  # 恢复累积的实际损失值
				))
				logging.info(f'Backward time: {backward_time:.6f} seconds')
				# 重置累积损失
				accumulated_loss = 0.0

			


def eval(model, val_loader,epoch):
	model.eval()
	total_qa = 0
	total_match=0
	correct_qa = 0
	correct_match=0
	with torch.no_grad():
		for batch_idx, sample in enumerate(val_loader):
			audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

			preds_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)

			_, predicted = torch.max(preds_qa.data, 1)
			total_qa += preds_qa.size(0)
			correct_qa += (predicted == target).sum().item()

	logging.info('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
	# writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

	return 100 * correct_qa / total_qa


def test(model, val_loader):
	model.eval()
	total = 0
	correct = 0
	samples = json.load(open('/mnt/sda/shenhao/code/LAVISH/AVQA/data/json/avqa-test.json', 'r'))
	A_count = []
	A_cmp = []
	V_count = []
	V_loc = []
	AV_ext = []
	AV_count = []
	AV_loc = []
	AV_cmp = []
	AV_temp = []
	with torch.no_grad():
		for batch_idx, sample in enumerate(val_loader):
			audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

			preds_qa,out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)
			preds = preds_qa
			_, predicted = torch.max(preds.data, 1)

			total += preds.size(0)
			correct += (predicted == target).sum().item()

			x = samples[batch_idx]
			type =ast.literal_eval(x['type'])
			if type[0] == 'Audio':
				if type[1] == 'Counting':
					A_count.append((predicted == target).sum().item())
				elif type[1] == 'Comparative':
					A_cmp.append((predicted == target).sum().item())
			elif type[0] == 'Visual':
				if type[1] == 'Counting':
					V_count.append((predicted == target).sum().item())
				elif type[1] == 'Location':
					V_loc.append((predicted == target).sum().item())
			elif type[0] == 'Audio-Visual':
				if type[1] == 'Existential':
					AV_ext.append((predicted == target).sum().item())
				elif type[1] == 'Counting':
					AV_count.append((predicted == target).sum().item())
				elif type[1] == 'Location':
					AV_loc.append((predicted == target).sum().item())
				elif type[1] == 'Comparative':
					AV_cmp.append((predicted == target).sum().item())
				elif type[1] == 'Temporal':
					AV_temp.append((predicted == target).sum().item())

	logging.info('Audio Counting Accuracy: %.2f %%' % (
			100 * sum(A_count)/len(A_count)))
	logging.info('Audio Cmp Accuracy: %.2f %%' % (
			100 * sum(A_cmp) / len(A_cmp)))
	logging.info('Audio Accuracy: %.2f %%' % (
			100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
	logging.info('Visual Counting Accuracy: %.2f %%' % (
			100 * sum(V_count) / len(V_count)))
	logging.info('Visual Loc Accuracy: %.2f %%' % (
			100 * sum(V_loc) / len(V_loc)))
	logging.info('Visual Accuracy: %.2f %%' % (
			100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
	logging.info('AV Ext Accuracy: %.2f %%' % (
			100 * sum(AV_ext) / len(AV_ext)))
	logging.info('AV counting Accuracy: %.2f %%' % (
			100 * sum(AV_count) / len(AV_count)))
	logging.info('AV Loc Accuracy: %.2f %%' % (
			100 * sum(AV_loc) / len(AV_loc)))
	logging.info('AV Cmp Accuracy: %.2f %%' % (
			100 * sum(AV_cmp) / len(AV_cmp)))
	logging.info('AV Temporal Accuracy: %.2f %%' % (
			100 * sum(AV_temp) / len(AV_temp)))

	logging.info('AV Accuracy: %.2f %%' % (
			100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
				   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

	logging.info('Overall Accuracy: %.2f %%' % (
			100 * correct / total))

	return 100 * correct / total

def main():
	# Training settings
	if args.wandb:
		wandb.init(config=args, project="AVQA",name=args.model_name)

	logging.info(torch.cuda.is_available())
	logging.info(torch.__version__)
	torch.manual_seed(args.seed)

	if args.model == 'AVQA_Fusion_Net':
		model = AVQA_Fusion_Net(args)
		# print(model)
		model = nn.DataParallel(model)
		model = model.to('cuda')
	else:
		raise ('not recognized')

	if args.mode == 'train':
		train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
									transform=transforms.Compose([ToTensor()]), mode_flag='train')
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
		val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
									transform=transforms.Compose([ToTensor()]), mode_flag='test')
		val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)


		# ===================================== load pretrained model ===============================================
		####### concat model
		pretrained_file = "/mnt/sda/shenhao/code/LAVISH/AVQA/grounding_gen/models_grounding_gen/main_grounding_gen_best.pt"
		checkpoint = torch.load(pretrained_file)
		print("\n-------------- loading pretrained models --------------")
		model_dict = model.state_dict()
		tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
		tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
		pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
		pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in checkpoint.items() if k in tmp2}

		model_dict.update(pretrained_dict1) #利用预训练模型的参数，更新模型
		model_dict.update(pretrained_dict2) #利用预训练模型的参数，更新模型
		model.load_state_dict(model_dict)

		print("\n-------------- load pretrained models --------------")

		# ===================================== load pretrained model ===============================================

		param_group = []
		train_params = 0
		total_params = 0
		additional_params = 0
		for name, param in model.named_parameters():
			
			param.requires_grad = True
			### ---> compute params
			tmp = 1
			for num in param.shape:
				tmp *= num

			if 'ViT'in name or 'swin' in name or 'Resnet' in name:
				if 'norm' in name:
					param.requires_grad = bool(args.is_vit_ln)
					total_params += tmp
					train_params += tmp
				else:
					param.requires_grad = False
					total_params += tmp
				
			# ### <----
			
			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name)
			else:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
	
			if 'adapter_blocks' in name:
				param_group.append({"params": param, "lr":args.lr_block})
			else:
				param_group.append({"params": param, "lr":args.lr})
		print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
		print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
		print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))


		optimizer = optim.Adam(model.parameters(), lr=args.lr)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
		criterion = nn.CrossEntropyLoss()
		best_F = 0
		count = 0
		for epoch in range(1, args.epochs + 1):
			train(args, model, train_loader, optimizer, criterion, epoch=epoch)
			scheduler.step(epoch)
			# F = eval(model, val_loader, epoch)
			F = test(model, val_loader)
			count +=1
			if F >= best_F:
				count = 0
				best_F = F
				if args.wandb:
					wandb.log({"val-best": best_F})
				# 添加时间戳到模型文件名
				timestamp = datetime.now().strftime("%Y%m%d_")
				model_save_path = args.model_save_dir + timestamp + args.checkpoint + ".pt"
				torch.save(model.state_dict(), model_save_path)
			if count == args.early_stop:
				exit()

	else:
		test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
								   transform=transforms.Compose([ToTensor()]), mode_flag='test')
		print(test_dataset.__len__())
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
		model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		test(model, test_loader)


if __name__ == '__main__':
	main()


