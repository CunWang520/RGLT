from torch_geometric.datasets import Planetoid, Amazon
import pickle as pkl

class NCDataset(object):
    def __init__(self, name):

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
def load_synthetic_dataset(data_dir, name, lang, gen_model='gcn'):
    dataset = NCDataset(lang)

    assert lang in range(0, 10), 'Invalid dataset'

    if name == 'cora':
        node_feat, y = pkl.load(open('{}/Planetoid/cora/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Planetoid(root='{}/Planetoid'.format(data_dir),
                              name='cora')
    elif name == 'amazon-photo':
        node_feat, y = pkl.load(open('{}/Amazon/Photo/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Amazon(root='{}/Amazon'.format(data_dir),
                                  name='Photo')
    data = torch_dataset[0]

    edge_index = data.edge_index
    # label = data.y
    label = y
    num_nodes = node_feat.size(0)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label

    return dataset

def load_nc_dataset(data_dir, dataname, sub_dataname='', gen_model='gcn'):
    """ Loader for NCDataset
        Returns NCDataset
    """
    if dataname in  ('cora', 'amazon-photo'):
        dataset = load_synthetic_dataset(data_dir, dataname, sub_dataname, gen_model)
    else:
        raise ValueError('Invalid dataname')
    return dataset
