import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

import argparse

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 64)
        #self.lr = torch.nn.Linear(256, 64)

    def forward(self, pos_edge_index, neg_edge_index):

        x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
        x = F.relu(self.conv2(x, data.train_pos_edge_index))
        embed = x

        #x = F.relu(self.conv2(x, data.train_pos_edge_index))
        x = self.conv3(x, data.train_pos_edge_index)
        #embed = x
        #x = self.lr(x)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j), embed

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    optimizer.zero_grad()

    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=pos_edge_index.size(1))

    link_logits, embed = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        outputs, embed = model(pos_edge_index, neg_edge_index)
        link_probs = torch.sigmoid(outputs)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--epoch', type=int, default=101,
                        help="set epoch")
    parser.add_argument('--seed', type=int, required=True,
                        help="set seed")
    parser.add_argument('--first_seed', type=int, default=666,
                        help="set first seed")
    parser.add_argument('--train_cls', action="store_true", default=False,
                        help="use pre-trained embedding to train cls")
    parser.add_argument('--fix_cls', action="store_true", default=False,
                        help="fix cls to test embedding")                        
    parser.add_argument('--load_row0_cls', action="store_true", default=False,
                        help="load row 0's cls weight to train another set of embedding")
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    # Train/validation/test
    #data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # loading row 0's cls
    if args.load_row0_cls:
        pretrained_model = Net().to(device)
        pretrained_model.load_state_dict(torch.load('./model/link_model_seed{}.pkl'.format(args.first_seed)))
        model.conv3 = pretrained_model.conv3
        #model.lr = pretrained_model.lr

        for param in model.conv3.parameters():
            param.requires_grad = False
    

    best_val_perf = test_perf = 0
    for epoch in range(1, args.epoch):
        train_loss = train()
        val_perf, tmp_test_perf = test()
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, best_val_perf, test_perf))

    torch.save(model.state_dict(), './model/link_model_seed{}.pkl'.format(seed))

    # collect embedding
    if args.train_cls:
        model.eval()
        
        x, pos_edge_index = data.x, data.train_pos_edge_index

        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                        num_nodes=x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1))

        _, embed = model(pos_edge_index, neg_edge_index)
        '''
        pos_edge_index = data.train_pos_edge_index
        neg_edge_index = data.train_neg_edge_index
        '''

        print(embed.shape)

        embed = embed[data.train_mask].detach().numpy()
        labels = data.y[data.train_mask]

        print(embed.shape, labels.shape)

        clf = SVC(random_state=seed)
        clf.fit(embed, labels)

        with open('./model/svm_seed_{}.pkl'.format(args.seed), 'wb') as f:
            pickle.dump(clf, f)

        # evaluate in testset
        prefix = "test"
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        _, embed = model(pos_edge_index, neg_edge_index)

        #predict = clf.predict(embed)
        test_embed = embed[data.test_mask].detach().numpy()
        test_labels = data.y[data.test_mask]

        print(test_embed.shape, test_labels.shape)
        print("Seed {}, Train SVC: test accu = {}".format(args.seed, clf.score(test_embed, test_labels)))
    
    if args.fix_cls:
        model.eval()

        with open('./model/svm_seed_{}.pkl'.format(args.first_seed), 'rb') as f:
            clf = pickle.load(f)

        # evaluate in testset
        prefix = "test"
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        _, embed = model(pos_edge_index, neg_edge_index)

        #predict = clf.predict(embed)
        test_embed = embed[data.test_mask].detach().numpy()
        test_labels = data.y[data.test_mask]

        print(test_embed.shape, test_labels.shape)
        if args.load_row0_cls:
            print("Seed {}, with fix cls, test SVC: test accu = {}".format(args.seed, clf.score(test_embed, test_labels)))
        else:
            print("Seed {}, w/o fix cls, test SVC: test accu = {}".format(args.seed, clf.score(test_embed, test_labels)))




