import os
import random
import argparse
import wandb

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import net as net
from utils import load_data
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
import warnings
from ood.ood_dataset import load_nc_dataset
from ood.ood_graph_editor import Graph_Editer

warnings.filterwarnings('ignore')
def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    return (y_true == y_pred).sum() / y_true.shape[0]
def get_dataset(dataset, sub_dataset=None, gen_model=None,args=None):
    ### Load and preprocess data ###
    if dataset == 'cora':
        dataset = load_nc_dataset(args['data_dir'], 'cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset = load_nc_dataset(args['data_dir'], 'amazon-photo', sub_dataset, gen_model)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']
    return dataset

def run_fix_mask(args, seed, rewind_weight_mask):
    pruning.setup_seed(seed)
    if args['dataset']=='cora':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = 'gcn'
        dataset_tr = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model,args=args)
        dataset_val = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model,args=args)
        datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model,args=args) for i in range(len(te_subs))]
    elif args['dataset'] == 'amazon-photo':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = 'gcn'
        dataset_tr = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model,args=args)
        dataset_val = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model,args=args)
        datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model,args=args) for i in range(len(te_subs))]
    print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
    print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
    for i in range(len(te_subs)):
        dataset_te = datasets_te[i]
        print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")


    # adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])

    features, labels = dataset_tr.graph['node_feat'].cuda(), dataset_tr.label.cuda()

    # adj = coo_matrix(
    #     (np.ones(len(dataset_tr.graph['edge_index'][1])), (dataset_tr.graph['edge_index'][0].numpy(), dataset_tr.graph['edge_index'][1].numpy())),
    #     shape=(dataset_tr.graph['num_nodes'], dataset_tr.graph['num_nodes']))#dataset_tr.graph['edge_index']
    #
    # # node_num = features.size()[0]
    # # class_num = labels.numpy().max() + 1
    # adj = adj.toarray()
    # adj = torch.from_numpy(adj)
    adj = to_dense_adj(dataset_tr.graph['edge_index'])[0].to(torch.int)
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()


    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    net_gcn.load_state_dict(rewind_weight_mask)

    adj_spar, wei_spar = pruning.print_sparsity(net_gcn)

    for name, param in net_gcn.named_parameters():
        # zeros = torch.zeros_like(sparse_adj)
        # ones = torch.ones_like(sparse_adj)
        # mask = torch.where(sparse_adj != 0, ones, zeros)
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_avg_acc=0
    best_acc_tests=[]

    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}

    for epoch in range(args['epochs'] ):
        beta = 1 * args["beta"] * epoch / args['epochs'] + args["beta"] * (1 - epoch / args['epochs'] )
        for m in range(args['T']):
            Loss, Log_p = [], 0
            #for k in range(args['K']):

            output = net_gcn(features, adj)
            labels = labels.squeeze(dim=-1)
            Loss = loss_func(output, labels)
            #Loss.append(loss.view(-1))

            #Loss = torch.cat(Loss, dim=0)
            #Var, Mean = torch.var_mean(Loss)
            #outer_loss = Var + beta * Mean


            optimizer.zero_grad()

            Loss.backward()
            optimizer.step()


        with torch.no_grad():
            features, labels = dataset_val.graph['node_feat'].cuda(), dataset_val.label.cuda()
            # adj = coo_matrix(
            #     (np.ones(len(dataset_val.graph['edge_index'][1])),
            #      (dataset_val.graph['edge_index'][0].numpy(), dataset_val.graph['edge_index'][1].numpy())),
            #     shape=(
            #     dataset_val.graph['num_nodes'], dataset_val.graph['num_nodes']))  # dataset_tr.graph['edge_index']
            # adj = torch.from_numpy(adj.toarray()).cuda()
            adj = to_dense_adj(dataset_val.graph['edge_index'])[0].to(torch.int).cuda()
            output = net_gcn(features, adj, val_test=True)
            # acc_val = f1_score(labels.cpu().numpy(), output.cpu().numpy().argmax(axis=1),
            #                    average='micro')
            acc_val = eval_acc(labels, output)
            acc_tests = []
            for i, dataset in enumerate(datasets_te):
                features, labels = dataset.graph['node_feat'].cuda(), dataset.label.cuda()
                # adj = coo_matrix(
                #     (np.ones(len(dataset.graph['edge_index'][1])),
                #      (dataset.graph['edge_index'][0].numpy(), dataset.graph['edge_index'][1].numpy())),
                #     shape=(dataset.graph['num_nodes'], dataset.graph['num_nodes']))  # dataset_tr.graph['edge_index']
                # adj = torch.from_numpy(adj.toarray()).cuda()
                adj = to_dense_adj(dataset.graph['edge_index'])[0].to(torch.int).cuda()
                output = net_gcn(features, adj, val_test=True)
                # acc_test = f1_score(labels.cpu().numpy(), output.cpu().numpy().argmax(axis=1),
                #                 average='micro')
                acc_test = eval_acc(labels, output)
                acc_tests.append(acc_test)

            if np.mean(acc_tests)>best_avg_acc:
                best_avg_acc = np.mean(acc_tests)
                best_acc_tests = acc_tests
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch


        print("(Fix Mask) Epoch:[{}] Val:[{:.2f}] | Final Val:[{:.2f}] at Epoch:[{}] | Best avg Test:[{:.2f}] "
              .format(epoch, acc_val * 100,
                      best_val_acc['val_acc'] * 100,
                      best_val_acc['epoch'],best_avg_acc*100))

        for index, item in enumerate(acc_tests):
            print('test ' + str(index), item * 100)

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch'],best_acc_tests, adj_spar, wei_spar


def run_get_mask(args, seed, imp_num, rewind_weight_mask=None):
    pruning.setup_seed(seed)
    ####################################################################################################################
    if args['dataset']=='cora':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = 'gcn'
        dataset_tr = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model,args=args)
        dataset_val = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model,args=args)
        datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model,args=args) for i in range(len(te_subs))]
    elif args['dataset'] == 'amazon-photo':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = 'gcn'
        dataset_tr = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model,args=args)
        dataset_val = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model,args=args)
        datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model,args=args) for i in range(len(te_subs))]
    print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
    print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
    for i in range(len(te_subs)):
        dataset_te = datasets_te[i]
        print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")


    # adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])

    features, labels = dataset_tr.graph['node_feat'].cuda(), dataset_tr.label.cuda()

    # adj = coo_matrix(
    #     (np.ones(len(dataset_tr.graph['edge_index'][1])), (dataset_tr.graph['edge_index'][0].numpy(), dataset_tr.graph['edge_index'][1].numpy())),
    #     shape=(dataset_tr.graph['num_nodes'], dataset_tr.graph['num_nodes']))#dataset_tr.graph['edge_index']
    #
    #
    #
    # # node_num = features.size()[0]
    # # class_num = labels.numpy().max() + 1
    #
    # adj = adj.toarray()
    # # print('adj',adj,adj.shape)
    # adj = torch.from_numpy(adj)
    adj = to_dense_adj(dataset_tr.graph['edge_index'])[0].to(torch.int)

    # print('adj',adj,adj.shape)

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            print(name)

    if args['weight_dir']:
        print("load : {}".format(args['weight_dir']))
        encoder_weight = {}
        cl_ckpt = torch.load(args['weight_dir'], map_location='cuda')
        encoder_weight['weight_orig_weight'] = cl_ckpt['gcn.fc.weight']
        ori_state_dict = net_gcn.net_layer[0].state_dict()
        ori_state_dict.update(encoder_weight)
        net_gcn.net_layer[0].load_state_dict(ori_state_dict)

    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
        if not args['rewind_soft_mask'] or args['init_soft_mask_type'] == 'all_one':
            pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
        adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    else:
        pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_avg_acc=0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    adj_grad = torch.zeros_like(net_gcn.adj_mask1_train).cpu()
    wei_01_grad = torch.zeros_like(net_gcn.net_layer[0].weight_mask_train).cpu()
    wei_02_grad = torch.zeros_like(net_gcn.net_layer[1].weight_mask_train).cpu()
    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        labels = labels.squeeze(dim=-1)
        loss = loss_func(output, labels)
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args)  # l1 norm

        adj_grad = adj_grad + net_gcn.adj_mask1_train.grad.clone().cpu()/args['total_epoch']
        wei_01_grad = wei_01_grad + net_gcn.net_layer[0].weight_mask_train.grad.clone().cpu()/args['total_epoch']
        wei_02_grad = wei_02_grad + net_gcn.net_layer[1].weight_mask_train.grad.clone().cpu()/args['total_epoch']

        optimizer.step()
        with torch.no_grad():
            features, labels = dataset_val.graph['node_feat'].cuda(), dataset_val.label.cuda()
            # adj = coo_matrix(
            #     (np.ones(len(dataset_val.graph['edge_index'][1])),
            #      (dataset_val.graph['edge_index'][0].numpy(), dataset_val.graph['edge_index'][1].numpy())),
            #     shape=(dataset_val.graph['num_nodes'], dataset_val.graph['num_nodes']))  # dataset_tr.graph['edge_index']
            # adj = torch.from_numpy(adj.toarray()).cuda()
            adj = to_dense_adj(dataset_val.graph['edge_index'])[0].to(torch.int).cuda()
            output = net_gcn(features, adj, val_test=True)
            # acc_val = f1_score(labels.cpu().numpy(), output.cpu().numpy().argmax(axis=1),
            #                    average='micro')
            acc_val = eval_acc(labels, output)
            acc_tests = []
            for i, dataset in enumerate(datasets_te):
                features, labels = dataset.graph['node_feat'].cuda(), dataset.label.cuda()
                # adj = coo_matrix(
                #     (np.ones(len(dataset.graph['edge_index'][1])),
                #      (dataset.graph['edge_index'][0].numpy(), dataset.graph['edge_index'][1].numpy())),
                #     shape=(dataset.graph['num_nodes'], dataset.graph['num_nodes']))  # dataset_tr.graph['edge_index']
                # adj = torch.from_numpy(adj.toarray()).cuda()
                adj = to_dense_adj(dataset.graph['edge_index'])[0].to(torch.int).cuda()
                output = net_gcn(features, adj, val_test=True)
                # acc_test = f1_score(labels.cpu().numpy(), output.cpu().numpy().argmax(axis=1),
                #                 average='micro')
                acc_test = eval_acc(labels, output)
                acc_tests.append(acc_test)
            if np.mean(acc_tests)>best_avg_acc:
                best_avg_acc = np.mean(acc_tests)
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_epoch_mask = pruning.get_final_mask_epoch02(net_gcn,adj_grad,wei_01_grad,wei_02_grad, adj_percent=args['pruning_percent_adj'],
                                                               wei_percent=args['pruning_percent_wei'])

            print("(Get Mask) Epoch:[{}] Val:[{:.2f}]  | Best Val:[{:.2f}]  at Epoch:[{}] | Best avg Test:[{:.2f}] "
                  .format(epoch, acc_val * 100,
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['epoch'],best_avg_acc*100))
            for index,item in enumerate(acc_tests):
                print('test '+str(index),item*100)

    return best_epoch_mask, rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--exp_name', type=str, default='gcn_no_editor')
    parser.add_argument('--s1', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--total_epoch', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)#epochs
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--weight_dir', type=str, default='')
    parser.add_argument('--rewind_soft_mask', action='store_true')
    parser.add_argument('--init_soft_mask_type', type=str, default='all_one', help='all_one, kaiming, normal, uniform')
    ###### Others settings #######
    parser.add_argument('--data_dir', type=str, default='../ood_data')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[1443, 512, 9])
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_a', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num_sample', type=int, default=1)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    # wandb.init(project="gnn_prune_ood", name=args['exp_name'] + '_' + args['dataset'])
    seed_dict = {'cora': 2377, 'citeseer': 4428, 'pubmed': 3333,'amazon-photo':1234}
    seed = seed_dict[args['dataset']]
    rewind_weight = None
    val = []
    test1 = []
    test2 = []
    test3 = []
    test4 = []
    test5 = []
    test6 = []
    test7 = []
    test8 = []
    best_avg_test = []
    Adj = []
    Wei = []
    for p in range(20):
        wandb_log = {}
        final_mask_dict, rewind_weight = run_get_mask(args, seed, p, rewind_weight)

        rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
        rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
        rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
        rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']

        best_acc_val, final_acc_test, final_epoch_list, best_acc_tests, adj_spar, wei_spar = run_fix_mask(args, seed,
                                                                                                          rewind_weight)
        wandb_log['Best Val'] = best_acc_val * 100
        wandb_log['Final Test Acc'] = final_acc_test * 100
        wandb_log['epoch'] = final_epoch_list
        wandb_log['Adj'] = adj_spar
        wandb_log['Wei'] = wei_spar
        wandb_log['test 1'] = best_acc_tests[0] * 100
        wandb_log['test 2'] = best_acc_tests[1] * 100
        wandb_log['test 3'] = best_acc_tests[2] * 100
        wandb_log['test 4'] = best_acc_tests[3] * 100
        wandb_log['test 5'] = best_acc_tests[4] * 100
        wandb_log['test 6'] = best_acc_tests[5] * 100
        wandb_log['test 7'] = best_acc_tests[6] * 100
        wandb_log['test 8'] = best_acc_tests[7] * 100
        wandb_log['Best Avg Test'] = np.mean(best_acc_tests) * 100
        print("=" * 120)
        print(
            "syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
            .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar, wei_spar))
        print("=" * 120)
        val.append(wandb_log['Best Val'])
        test1.append(wandb_log['test 1'])
        test2.append(wandb_log['test 2'])
        test3.append(wandb_log['test 3'])
        test4.append(wandb_log['test 4'])
        test5.append(wandb_log['test 5'])
        test6.append(wandb_log['test 6'])
        test7.append(wandb_log['test 7'])
        test8.append(wandb_log['test 8'])
        best_avg_test.append(wandb_log['Best Avg Test'])
        Adj.append(wandb_log['Adj'])
        Wei.append(wandb_log['Wei'])
        # wandb.log(wandb_log)
        Adj = [(100 - x) / 100 for x in Adj]
    Wei = [(100 - x) / 100 for x in Wei]
    val = [x / 100 for x in val]
    test1 = [x / 100 for x in test1]
    test2 = [x / 100 for x in test2]
    test3 = [x / 100 for x in test3]
    test4 = [x / 100 for x in test4]
    test5 = [x / 100 for x in test5]
    test6 = [x / 100 for x in test6]
    test7 = [x / 100 for x in test7]
    test8 = [x / 100 for x in test8]
    print('val', val)
    print('test1', test1)
    print('test2', test2)
    print('test3', test3)
    print('test4', test4)
    print('test5', test5)
    print('test6', test6)
    print('test7', test7)
    print('test8', test8)
    print('best_avg_test',best_avg_test)
    print('Adj', Adj)
    print('Wei', Wei)