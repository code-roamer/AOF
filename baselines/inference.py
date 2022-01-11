"""Test the victim models"""
import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import ModelNet40Attack, ModelNet40Transfer, load_data
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import AverageMeter, str2bool, set_seed
from config import BEST_WEIGHTS
from config import MAX_TEST_BATCH as BATCH_SIZE
from config import MAX_DUP_TEST_BATCH as DUP_BATCH_SIZE

def merge(data_root, prefix):
    ori_data_lst = []
    adv_data_lst = []
    label_lst = []
    save_name = prefix+"-concat.npz"
    if os.path.exists(os.path.join(data_root, save_name)):
        return os.path.join(data_root, save_name)
    for file in os.listdir(data_root):
        if file.startswith(prefix):
            file_path = os.path.join(data_root, file)
            ori_data, adv_data, label = \
                load_data(file_path, partition='transfer')
            ori_data_lst.append(ori_data)
            adv_data_lst.append(adv_data)
            label_lst.append(label)
    all_ori_pc = np.concatenate(ori_data_lst, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(adv_data_lst, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(label_lst, axis=0)  # [num_data]

    np.savez(os.path.join(data_root, save_name),
             ori_pc=all_ori_pc.astype(np.float32),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8))
    return os.path.join(data_root, save_name)


def get_model_name(npz_path):
    """Get the victim model name from npz file path."""
    if 'dgcnn' in npz_path.lower():
        return 'dgcnn'
    if 'pointconv' in npz_path.lower():
        return 'pointconv'
    if 'pointnet2' in npz_path.lower():
        return 'pointnet2'
    if 'pointnet' in npz_path.lower():
        return 'pointnet'
    print('Victim model not recognized!')
    exit(-1)


def test_target():
    """Target test mode.
    Show both classification accuracy and target success rate.
    """
    model.eval()
    acc_save = AverageMeter()
    success_save = AverageMeter()
    with torch.no_grad():
        for data, label, target in test_loader:
            data, label, target = \
                data.float().cuda(), label.long().cuda(), target.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if args.model.lower() == 'pointnet':
                logits, _, _ = model(data)
            else:
                logits = model(data)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)
            success = (preds == target).sum().float() / float(batch_size)
            success_save.update(success.item(), batch_size)

    print('Overall accuracy: {:.4f}, '
          'attack success rate: {:.4f}'.
          format(acc_save.avg, success_save.avg))


def test_normal():
    """Normal test mode.
    Test on all data.
    """
    model.eval()
    at_num, at_denom = 0, 0

    num, denom = 0, 0
    with torch.no_grad():
        for ori_data, adv_data, label in test_loader:
            ori_data, adv_data, label = \
                ori_data.float().cuda(), adv_data.float().cuda(), label.long().cuda()
            # to [B, 3, N] point cloud
            ori_data = ori_data.transpose(1, 2).contiguous()
            adv_data = adv_data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if args.model.lower() == 'pointnet':
                logits, _, _ = model(ori_data)
                adv_logits, _, _ = model(adv_data)
            else:
                logits = model(ori_data)
                adv_logits = model(adv_data)
            ori_preds = torch.argmax(logits, dim=-1)
            adv_preds = torch.argmax(adv_logits, dim=-1)
            mask_ori = (ori_preds == label)
            mask_adv = (adv_preds == label)
            at_denom += mask_ori.sum().float()
            at_num += mask_ori.sum().float() - (mask_ori * mask_adv).sum().float()
            denom += float(batch_size)
            num += mask_adv.sum().float()

    print('Overall attack success rate: {:.4f}'.format(at_num / (at_denom + 1e-9)))
    print('Overall accuracy: {:.4f}'.format(num / (denom + 1e-9)))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='')
    parser.add_argument('--prefix', type=str,
                        default='')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'target'],
                        help='Testing mode')
    parser.add_argument('--model', type=str, default='', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', ''],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40','ori_mn40'])
    parser.add_argument('--normalize_pc', type=str2bool, default=False,
                        help='normalize in dataloader')
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch, use config if not specified')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Model weight to load, use config if not specified')


    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()

    # victim model
    if not args.model:
        args.model = get_model_name(args.data_root)

    # random seed
    set_seed(1)

    # in case adding attack
    if 'add' in args.data_root.lower():
        # we add 512 points in adding attacks
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 512
        elif args.num_points == 1024 + 512:
            num_points = 1024
    elif 'cluster' in args.data_root.lower():
        # we add 3*32=96 points in adding cluster attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 32
        elif args.num_points == 1024 + 3 * 32:
            num_points = 1024
    elif 'object' in args.data_root.lower():
        # we add 3*64=192 points in adding object attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 64
        elif args.num_points == 1024 + 3 * 64:
            num_points = 1024
    else:
        num_points = args.num_points

    # determine the weight to use
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][num_points]
    BATCH_SIZE = BATCH_SIZE[num_points]
    DUP_BATCH_SIZE = DUP_BATCH_SIZE[num_points]
    if args.batch_size == -1:  # automatic assign
        args.batch_size = BATCH_SIZE[args.model]
    # add point attack has more points in each point cloud
    if 'ADD' in args.data_root:
        args.batch_size = int(args.batch_size / 1.5)
    # sor processed point cloud has different points in each
    # so batch size only can be 1
    if 'sor' in args.data_root:
        args.batch_size = 1

    # enable cudnn benchmark
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

    model = nn.DataParallel(model).cuda()

    # load model weight
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(BEST_WEIGHTS[args.model]))

    data_path = merge(args.data_root, args.prefix)
    # prepare data
    if args.mode == 'target':
        test_set = ModelNet40Attack(data_path, num_points=args.num_points,
                                    normalize=args.normalize_pc)
    else:
        test_set = ModelNet40Transfer(data_path, num_points=args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=8,
                             pin_memory=True, drop_last=False)

    # test
    if args.mode == 'normal':
        test_normal()
    else:
        test_target()
