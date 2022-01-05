"""Targeted kNN attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
sys.path.append('./')

from config import BEST_WEIGHTS
from config import MAX_KNN_BATCH as BATCH_SIZE
from dataset import ModelNetDataLoader
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import CWKNN, CWUKNN
from attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from attack import ChamferkNNDist
from attack import ProjectInnerClipLinf


def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    num = 0
    at_num, trans_num, total_num = 0, 0, 0
    for pc, label in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)

        # attack!
        best_pc, success_num = attacker.attack(pc, label)
        adv_pc = torch.tensor(best_pc.transpose(0, 2, 1)).to(pc)
        if args.model.lower() == 'pointnet':
            logits, _, _ = model(adv_pc)
        else:
            logits = model(adv_pc)
        preds = torch.argmax(logits, dim=-1)

        if args.trans_model.lower() == 'pointnet':
            trans_logits, _, _ = trans_model(adv_pc)
        else:
            trans_logits = trans_model(adv_pc)
        trans_preds = torch.argmax(trans_logits, dim=-1)

        at_num += (preds != label).sum().item()
        trans_num += (trans_preds != label).sum().item()
        
        total_num += args.batch_size

        
        print(f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}")
        # np.save(f'visual/vis_knn.npy', best_pc.transpose(0, 2, 1))
        # break

        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())

    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, num


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='official_data/modelnet40_normal_resampled')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--trans_model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40', 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=15.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr', type=float, default=1e-3,
                        help='lr in CW optimization')
    parser.add_argument('--num_iter', type=int, default=2500, metavar='N',
                        help='Number of iterations in each search step')

    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()

    # build transfer model
    if args.trans_model.lower() == 'dgcnn':
        trans_model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.trans_model.lower() == 'pointnet':
        trans_model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.trans_model.lower() == 'pointnet2':
        trans_model = PointNet2ClsSsg(num_classes=40)
    elif args.trans_model.lower() == 'pointconv':
        trans_model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)
    trans_model = torch.nn.DataParallel(trans_model).cuda()
    trans_model.load_state_dict(torch.load(BEST_WEIGHTS[args.trans_model]))
    trans_model.eval()

    # setup attack settings
    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    # hyper-parameters from their official tensorflow code
    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)
    clip_func = ProjectInnerClipLinf(budget=0.18)
    attacker = CWKNN(model, adv_func, dist_func, clip_func,
                     attack_lr=args.attack_lr,
                     num_iter=args.num_iter)

    # attack
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False)

    # run attack
    attacked_data, real_label, success_num = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/kNN'.\
        format(args.dataset, args.num_points)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.adv_func == 'logits':
        args.adv_func = 'logits_kappa={}'.format(args.kappa)
    save_name = 'kNN-{}-{}-success_{:.4f}.npz'.\
        format(args.model, args.adv_func,
               success_rate)
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8))
