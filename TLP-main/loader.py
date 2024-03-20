import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Subset

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data_with_features(link_path, node_features_path, ratio=1e5, max_thres=1e5):
    """加载带有节点特征的数据集"""
    print('Loading {} link dataset...'.format(link_path))
    print('Loading {} node features dataset...'.format(node_features_path))
    
    # 加载邻接矩阵
    adj_reg = np.load(link_path)
    adj_reg = adj_reg / ratio
    # adj_reg[np.abs(adj_reg) < 1] = 0
    adj_reg = np.clip(adj_reg, 0, max_thres)
    adj_reg = np.array(adj_reg, dtype=np.float32)
    # 加载节点特征
    feature = np.load(node_features_path)
    feature = normalize(feature)
    
    return adj_reg, feature

class TradeLinkDataset(Dataset):
    def __init__(self, adj_reg, node_features, window_size):
        self.adj_reg = adj_reg
        self.node_features = node_features
        self.window_size = window_size
        self.num = self.adj_reg.shape[0] - self.window_size
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # start_idx = idx
        # end_idx = idx + self.window_size
        # input_reg = self.adj_reg[start_idx:end_idx]  # 获取历史窗口期的邻接矩阵
        # target_reg = self.adj_reg[end_idx]  # 获取目标邻接矩阵
        # node_features = self.node_features[start_idx:end_idx] # 获取历史窗口期的节点特征
        
        # return {
        #     'node_features': node_features,
        #     'input_reg': input_reg,
        #     'target_reg': target_reg
        # }
        return self.adj_reg[idx:idx+self.window_size], self.adj_reg[idx+self.window_size]


def split_dataset(dataset, train_ratio=0.8):
        train_size = int(train_ratio * len(dataset))
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(dataset)))
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        return train_dataset, val_dataset

if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    batch_size = 1
    num_workers = 0
    
    # 定义数据路径和参数
    link_path = './data/Amount.npy'
    node_features_path = './data/population_2000-2022.npy'
    window_size = 5  # 窗口期大小
    
    # 加载数据
    adj_reg, node_features = load_data_with_features(link_path, node_features_path)
    
    # 创建数据集
    dataset = TradeLinkDataset(adj_reg, node_features, window_size)
    
    # 分割数据集为训练集和验证集
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)