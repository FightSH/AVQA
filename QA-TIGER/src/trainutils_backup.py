from typing import Dict, List, Tuple
from collections import defaultdict

import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import get_logger
from src.dataset import AVQA_dataset, qtype2idx
from src.utils import calculate_parameters
from src.models.net import QA_TIGER
from src.models.tspm import TSPM
from src.error_analyzer import ErrorAnalyzer, get_answer_from_idx

def train(cfg: dict,
          epoch: int,
          device: torch.device,
          train_loader: DataLoader,
          optimizer: Optimizer,
          criterion: nn.Module,
          model: nn.Module,
          writer: SummaryWriter = None,
          error_analyzer: ErrorAnalyzer = None,
          ):
    logger = get_logger()

    model.train()
    avg_meter = AverageMeter()
    tot_batch = len(train_loader) - 1

    epoch_time = time.time()
    for batch_idx, sample in enumerate(train_loader):
        start_time = time.time()

        reshaped_data = get_items(sample, device)
        optimizer.zero_grad()
        output = model(reshaped_data)

        loss = 0
        target = reshaped_data['label']
        ce_loss = criterion(output, target)
        loss += ce_loss
        losses = [('ce_loss', ce_loss)]
        
        # 记录训练错误（可选，通常只在验证和测试时记录）
        if error_analyzer is not None:
            with torch.no_grad():
                _, predicted = torch.max(output['out'].data, 1)
                probabilities = F.softmax(output['out'], dim=1)
                confidences, _ = torch.max(probabilities, 1)
                
                # 记录错误预测
                wrong_mask = (predicted != target)
                if wrong_mask.any():
                    wrong_indices = torch.nonzero(wrong_mask).squeeze()
                    if wrong_indices.dim() == 0:
                        wrong_indices = [wrong_indices.item()]
                    else:
                        wrong_indices = wrong_indices.tolist()
                    
                    for i in wrong_indices:
                        sample_info = train_loader.dataset.get_sample_info(sample['name'][i] if isinstance(sample['name'], list) else batch_idx * train_loader.batch_size + i)
                        predicted_answer = get_answer_from_idx(predicted[i].item(), train_loader.dataset.answer_to_ix)
                        
                        error_analyzer.record_error(
                            sample_info=sample_info,
                            predicted_answer=predicted_answer,
                            predicted_logits=output['out'][i],
                            confidence=confidences[i].item(),
                            epoch=epoch,
                            batch_idx=batch_idx
                        )
        
        # ...existing code for loss calculation and backward pass...
        for key in output:
            if 'bias_logits' in key and output[key] is not None:
                bias_loss = criterion(output[key], target)
                loss += bias_loss
                losses.append((key, bias_loss))
        losses.append(('total_loss', loss))
        loss.backward()
        optimizer.step()

        losses = gather_losses(epoch, batch_idx, tot_batch,
                               losses, writer, device)
        avg_meter.update(losses, step_n=1)

        if batch_idx % cfg.log_interval == 0 or batch_idx == len(train_loader) - 1:
            elapsed = time.time() - start_time
            logger.info(f'Train epoch: {epoch} [{batch_idx:>3d}/{tot_batch:>3d}] '
                       f'Loss: {avg_meter.get("total_loss"):.3f} '
                       f'Time: {elapsed:.2f}s')

        if cfg.debug and batch_idx == 10:
            break


def evaluate(cfg: dict,
             epoch: int,
             device: torch.device,
             val_loader: DataLoader,
             criterion: nn.Module,
             model: nn.Module,
             writer: SummaryWriter = None,
             error_analyzer: ErrorAnalyzer = None):
    global qtype2idx

    logger = get_logger()
    model.eval()

    loss = 0
    total, correct = 0, 0
    tot_tensor = torch.zeros(9, dtype=torch.long).to(device)
    correct_tensor = torch.zeros(9, dtype=torch.long).to(device)

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            reshaped_data = get_items(sample, device)
            qst_types = sample['type']
            target = reshaped_data['label']
            output = model(reshaped_data)
            _, predicted = torch.max(output['out'].data, 1)

            total += predicted.size(0)
            correct += (predicted == target).sum().item()
            loss += criterion(output, target) / len(val_loader)
            
            # 记录验证错误
            if error_analyzer is not None:
                probabilities = F.softmax(output['out'], dim=1)
                confidences, _ = torch.max(probabilities, 1)
                
                # 记录错误预测
                wrong_mask = (predicted != target)
                if wrong_mask.any():
                    wrong_indices = torch.nonzero(wrong_mask).squeeze()
                    if wrong_indices.dim() == 0:
                        wrong_indices = [wrong_indices.item()]
                    else:
                        wrong_indices = wrong_indices.tolist()
                    
                    for i in wrong_indices:
                        # 获取样本在数据集中的真实索引
                        dataset_idx = batch_idx * val_loader.batch_size + i
                        sample_info = val_loader.dataset.get_sample_info(dataset_idx)
                        predicted_answer = get_answer_from_idx(predicted[i].item(), val_loader.dataset.answer_to_ix)
                        
                        error_analyzer.record_error(
                            sample_info=sample_info,
                            predicted_answer=predicted_answer,
                            predicted_logits=output['out'][i],
                            confidence=confidences[i].item(),
                            epoch=epoch,
                            batch_idx=batch_idx
                        )

            for idx, (modal_type, qst_type) in enumerate(zip(qst_types[0], qst_types[1])):
                gather_idx = qtype2idx[modal_type][qst_type]
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()

            if cfg.debug and batch_idx == 10:
                break

            if batch_idx % cfg.log_interval == 0 or batch_idx == len(val_loader) - 1:
                logger.info(f'Valid progress: {batch_idx:3.0f}/{len(val_loader) - 1}')

    sync_processes()
    if dist.is_initialized():
        correct = torch.tensor(correct).to(device)
        total = torch.tensor(total).to(device)

        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        for idx in range(9):
            dist.all_reduce(tot_tensor[idx], op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor[idx], op=dist.ReduceOp.SUM)

    acc = correct / total * 100.
    loss = loss.item()
    if writer is not None:
        writer.add_scalar('valid/acc/Total', acc, epoch)

    for modality in ['Audio', 'Visual', 'Audio-Visual']:
        modality_corr = 0
        modality_tot = 0

        for qst_type in qtype2idx[modality]:
            corr = correct_tensor[qtype2idx[modality][qst_type]].item()
            tot = tot_tensor[qtype2idx[modality][qst_type]].item()

            modality_corr += corr
            modality_tot += tot
            value = corr / tot * 100.

            key = f'{modality}/{qst_type}'
            logger.info(f'Epoch {epoch} - {key:>24} accuracy: {value:.2f}({corr}/{tot})')
            if writer is not None:
                writer.add_scalar(f'valid/acc/{key}', corr / tot * 100., epoch)

        modality_acc = modality_corr / modality_tot * 100.
        logger.info(f'Epoch {epoch} - {modality:>24} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
        if writer is not None:
            writer.add_scalar(f'valid/acc/{modality}', modality_acc, epoch)
    key = 'Total'
    logger.info(f'Epoch {epoch} - {key:>24} accuracy: {acc:.2f}({correct}/{total})')

    return acc, loss

def test(cfg: dict,
         device: torch.device,
         val_loader: DataLoader,
         model: nn.Module,
         error_analyzer: ErrorAnalyzer = None):
    global qtype2idx

    logger = get_logger()
    model.eval()

    total, correct = 0, 0
    tot_tensor = torch.zeros(9, dtype=torch.long).to(device)
    correct_tensor = torch.zeros(9, dtype=torch.long).to(device)

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            reshaped_data = get_items(sample, device)

            qst_types = sample['type']
            target = reshaped_data['label']
            output = model(reshaped_data)
            _, predicted = torch.max(output['out'].data, 1)
            total += predicted.size(0)
            correct += (predicted == target).sum().item()
            
            # 记录测试错误
            if error_analyzer is not None:
                probabilities = F.softmax(output['out'], dim=1)
                confidences, _ = torch.max(probabilities, 1)
                
                # 记录错误预测
                wrong_mask = (predicted != target)
                if wrong_mask.any():
                    wrong_indices = torch.nonzero(wrong_mask).squeeze()
                    if wrong_indices.dim() == 0:
                        wrong_indices = [wrong_indices.item()]
                    else:
                        wrong_indices = wrong_indices.tolist()
                    
                    for i in wrong_indices:
                        # 获取样本在数据集中的真实索引
                        dataset_idx = batch_idx * val_loader.batch_size + i
                        sample_info = val_loader.dataset.get_sample_info(dataset_idx)
                        predicted_answer = get_answer_from_idx(predicted[i].item(), val_loader.dataset.answer_to_ix)
                        
                        error_analyzer.record_error(
                            sample_info=sample_info,
                            predicted_answer=predicted_answer,
                            predicted_logits=output['out'][i],
                            confidence=confidences[i].item(),
                            epoch=None,  # 测试时没有epoch
                            batch_idx=batch_idx
                        )
            
            for idx, (modal_type, qst_type) in enumerate(zip(qst_types[0], qst_types[1])):
                gather_idx = qtype2idx[modal_type][qst_type]
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()

            if cfg.debug and batch_idx == 10:
                break

            if batch_idx % cfg.log_interval == 0 or batch_idx == len(val_loader) - 1:
                logger.info(f'Test progress: {batch_idx:3.0f}/{len(val_loader) - 1}')

    sync_processes()
    if dist.is_initialized():
        correct = torch.tensor(correct).to(device)
        total = torch.tensor(total).to(device)

        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        for idx in range(9):
            dist.all_reduce(tot_tensor[idx], op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor[idx], op=dist.ReduceOp.SUM)

    acc = correct / total * 100.
    for modality in ['Audio', 'Visual', 'Audio-Visual']:
        modality_corr = 0
        modality_tot = 0

        for qst_type in qtype2idx[modality]:
            corr = correct_tensor[qtype2idx[modality][qst_type]].item()
            tot = tot_tensor[qtype2idx[modality][qst_type]].item()

            modality_corr += corr
            modality_tot += tot
            value = corr / tot * 100.

            key = f'{modality}/{qst_type}'
            logger.info(f'Test {key:>24} accuracy: {value:.2f}({corr}/{tot})')

        modality_acc = modality_corr / modality_tot * 100.
        logger.info(f'Test {modality:>24} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
    key = 'Total avg'
    logger.info(f'Test {key:>24} accuracy: {acc:.2f}({correct}/{total})')
    return acc