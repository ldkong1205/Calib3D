import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import init_default_scope
from mmengine import Config
from mmengine.runner import Runner, load_checkpoint
from torch import optim

from mmdet3d.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--calib_model', type=str, default='TempS', help='Calibration model')
    parser.add_argument('--num_class', type=int, default=20, help='number of class')
    parser.add_argument('--threshold', type=float, default=0.3, help='threshold')
    parser.add_argument('--ignore_index', type=int, default=19, help='ignored label')
    parser.add_argument('--val_freq', type=int, default=1500, help='number of iterations for a validation')
    args = parser.parse_args()
    return args


class Temperature_Scaling(nn.Module):

    def __init__(self):
        super(Temperature_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        temperature = self.temperature_single.expand(logits.size()).cuda()
        return logits / temperature


class Depth_Aware_Scaling(nn.Module):

    def __init__(self, threshold):
        super(Depth_Aware_Scaling, self).__init__()
        self.T1 = nn.Parameter(torch.ones(1))
        self.T2 = nn.Parameter(torch.ones(1)*0.9)

        self.k = nn.Parameter(torch.ones(1)*0.1)
        self.b = nn.Parameter(torch.zeros(1))

        self.alpha = 0.05
        self.threshold = threshold

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, logits, gt, xyz):
        if self.training:
            depth = torch.norm(xyz, p=2, dim=1).to(logits.device)

            ind = torch.argmax(logits, axis=1) == gt
            logits_pos, gt_pos, depth_pos = logits[ind], gt[ind], depth[ind]
            logits_neg, gt_neg, depth_neg = logits[~ind], gt[~ind], depth[~ind]

            start = np.random.randint(int(logits_pos.shape[0] * 1 / 3)) + 1
            logits = torch.cat((logits_neg, logits_pos[start:int(logits_pos.shape[0] / 2) + start]), 0)
            gt = torch.cat((gt_neg, gt_pos[start:int(logits_pos.shape[0] / 2) + start]), 0)
            depth = torch.cat((depth_neg, depth_pos[start:int(depth_pos.shape[0] / 2) + start]), 0)

            prob = self.softmax(logits)

            score = torch.sum(-prob * torch.log(prob), dim=-1)
            cond_ind = score < self.threshold

            cal_logits_1, cal_gt_1 = logits[cond_ind], gt[cond_ind]
            cal_logits_2, cal_gt_2 = logits[~cond_ind], gt[~cond_ind]

            # depth-aware scaling (DeptS)
            depth_coff = self.k * depth + self.b
            T1 = self.T1 * depth_coff[cond_ind].unsqueeze(dim=-1)
            T2 = self.T2 * depth_coff[~cond_ind].unsqueeze(dim=-1)

            cal_logits_1 = cal_logits_1 / T1
            cal_logits_2 = cal_logits_2 / T2

            cal_logits = torch.cat((cal_logits_1, cal_logits_2), 0)
            cal_gt = torch.cat((cal_gt_1, cal_gt_2), 0)

        else:
            prob = self.softmax(logits)

            score = torch.sum(-prob * torch.log(prob), dim=-1)
            cond_ind = score < self.threshold
            
            scaled_logits, scaled_gt = logits[cond_ind], gt[cond_ind]
            inference_logits, inference_gt = logits[~cond_ind], gt[~cond_ind]

            # depth-aware scaling (DeptS)
            depth = torch.norm(xyz, p=2, dim=1).to(logits.device)
            depth_coff = self.k * depth + self.b

            T1 = self.T1 * depth_coff[cond_ind].unsqueeze(dim=-1)
            T2 = self.T2 * depth_coff[~cond_ind].unsqueeze(dim=-1)

            scaled_logits = scaled_logits / T1
            inference_logits = inference_logits / T2

            cal_logits = torch.cat((scaled_logits, inference_logits), 0)
            cal_gt = torch.cat((scaled_gt, inference_gt), 0)
        
        return cal_logits, cal_gt
    

def calculate_ece(logits, labels, ignore_index, n_bins=10):
    labels = labels[:logits.shape[0]]
    valid_index = labels != ignore_index
    logits = logits[valid_index]
    labels = labels[valid_index]

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(
            bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin -
                             accuracy_in_bin) * prop_in_bin
    return ece.item()


if __name__ == '__main__':
    args = parse_args()
    init_default_scope('mmdet3d')
 
    cfg = Config.fromfile(args.config)
    train_dataloader = Runner.build_dataloader(cfg.train_dataloader)
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    if args.calib_model == 'TempS':
        calib_model = Temperature_Scaling()
    elif args.calib_model == 'DeptS':
        calib_model = Depth_Aware_Scaling(args.threshold)

    calib_model.cuda()
    nll_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    optimizer = optim.AdamW(calib_model.parameters(), lr=1e-3, weight_decay=1e-6)

    best_ece = 100

    for epoch in range(20):
        calib_model.train()
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            with torch.no_grad():
                results = model.test_step(data)
                logits = [result.pts_seg_logits.pts_seg_logits for result in results]
                logits = torch.cat(logits, dim=1)
                logits = logits.transpose(1, 0)
                labels = [result.gt_pts_seg.pts_semantic_mask for result in results]
                labels = torch.cat(labels)

            if args.calib_model == 'DeptS':
                points = data['inputs']['points']
                xyz = [point[:, :3] for point in points]
                xyz = torch.cat(xyz, dim=0)
                logits, labels = calib_model(logits, labels, xyz)
            else:
                logits = calib_model(logits)
            
            loss = torch.sum(nll_criterion(logits, labels))
            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print(f'epoch: [{epoch}]/[20] - iter: [{i}] / [{len(train_dataloader)}] - loss: {loss.item()}')

            if i % args.val_freq == 0 and i != 0:
                calib_model.eval()
                eces = []
                for i, data in enumerate(val_dataloader):

                    with torch.no_grad():
                        results = model.test_step(data)
                        logits = [result.pts_seg_logits.pts_seg_logits for result in results]
                        logits = torch.cat(logits, dim=1)
                        logits = logits.transpose(1, 0)
                        labels = [result.gt_pts_seg.pts_semantic_mask for result in results]
                        labels = torch.cat(labels)

                        if args.calib_model == 'DeptS':
                            points = data['inputs']['points']
                            xyz = [point[:, :3] for point in points]
                            xyz = torch.cat(xyz, dim=0)
                            logits, labels = calib_model(logits, labels, xyz)
                        else:
                            logits = calib_model(logits)
                    
                    ece = calculate_ece(logits, labels, args.ignore_index)
                    eces.append(ece)
                
                curr_ece = sum(eces) / len(eces)

                if curr_ece <= best_ece:
                    best_ece = curr_ece
                    torch.save(calib_model.state_dict(), f'{args.checkpoint}_{args.calib_model}.pth')
                
                print(f'ece: {curr_ece:.4f} | best_ece: {best_ece:.4f}')
                print("Validation complete")

                calib_model.train()
