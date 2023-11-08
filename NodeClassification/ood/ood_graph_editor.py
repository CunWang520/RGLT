from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph_Editer(nn.Module):
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k):
        Bk = self.B[k]
        # A = to_dense_adj(edge_index)[0].to(torch.int)
        A = edge_index.to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s] #对每一行进行无放回 采样 返回的是 索引地址
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        # edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return C, log_p


class Graph_Editer_Mask(nn.Module):
    def __init__(self, K, n, device, adj):
        super(Graph_Editer_Mask, self).__init__()
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(K,adj), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(K,adj), requires_grad=False)
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k,rate=0.15):
        # print('edge_index',edge_index[0,:])
        # print('self.adj_mask1_train[k].abs()',self.adj_mask1_train[k].abs())
        mask_y, mask_i = torch.sort(self.adj_mask1_train[k].view(-1))
        nonzero = torch.abs(edge_index.flatten()) > 0
        nonzero = len(torch.nonzero(nonzero))
        #print('nonzero',nonzero)
        mask_thre_index = int(nonzero * rate)
        #print('disturbance_num:',mask_thre_index)
        mask_thre = mask_y[mask_thre_index]
        # print('mask_thre',mask_thre)
        ones = torch.ones_like(self.adj_mask1_train[k])
        zeros = torch.zeros_like(self.adj_mask1_train[k])
        mask = torch.where(self.adj_mask1_train[k].abs() < mask_thre, ones, zeros)
        edge_index = edge_index + mask

        return edge_index

    def generate_adj_mask(self,K, input_adj):
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)  # .double()
        mask = torch.unsqueeze(mask, 0).repeat(K, 1, 1).float()
        #mask = mask + torch.rand_like(mask)
        mask = torch.rand_like(mask)
        return mask
        
        