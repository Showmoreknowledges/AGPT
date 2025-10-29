import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv as DGLGraphConv, HeteroGraphConv

from ..models.other_models import GCN


class NodeEncoder(nn.Module):
    """
    Handles encoding of features & PEs

    Also how to combine them
    """
    def __init__(
        self, 
        data,
        train_args,
        device="cuda"
    ):
        super().__init__()

        self.device = device
        self.dim = train_args['dim'] 
        init_dim = self.dim if 'emb' in data else data['x'].size(1)

        self.feat_drop = train_args.get('feat_drop', 0)
        self.use_relu = train_args.get('relu', True)

        self.use_relational_gnn = bool(
            train_args.get('use_relational_gnn', False) and data.get('hetero_graph') is not None
        )
        self.relation_agg = train_args.get('relation_agg', 'sum')

        self.input_proj = None
        self.rel_layers = None
        self.rel_norms = None
        self.hetero_graph = None
        self.primary_ntype = None

        if self.use_relational_gnn:
            self.hetero_graph = data['hetero_graph']
            self.primary_ntype = data.get('hetero_primary_ntype', self.hetero_graph.ntypes[0])
            if hasattr(self.hetero_graph, 'to'):
                self.hetero_graph = self.hetero_graph.to(device)

            if init_dim != self.dim:
                self.input_proj = nn.Linear(init_dim, self.dim)
                conv_in_dim = self.dim
            else:
                conv_in_dim = init_dim

            allow_zero = train_args.get('allow_zero_in_degree', True)
            self.rel_layers = nn.ModuleList()
            if train_args.get('layer_norm', False):
                self.rel_norms = nn.ModuleList()

            for _ in range(train_args['gnn_layers']):
                conv = HeteroGraphConv(
                    {
                        etype: DGLGraphConv(conv_in_dim, self.dim, allow_zero_in_degree=allow_zero)
                        for etype in self.hetero_graph.etypes
                    },
                    aggregate=self.relation_agg
                )
                self.rel_layers.append(conv)
                if self.rel_norms is not None:
                    self.rel_norms.append(nn.LayerNorm(self.dim))
                conv_in_dim = self.dim

            self.gnn_encoder = None
        else:
            self.gnn_encoder = GCN(init_dim, self.dim, self.dim, train_args['gnn_layers'],
                                   train_args.get('gnn_drop', 0), cached=train_args.get('gcn_cache'),
                                   residual=train_args['residual'], layer_norm=train_args['layer_norm'],
                                   relu=train_args['relu'])


    def forward(self, features, adj_t, test_set=False):
        """
        1. Transform all PEs
        2. Transform all node features
        3. Nodes + PEs
        """
        features = features.to(self.device)

        if self.input_proj is not None:
            features = self.input_proj(features)

        features = F.dropout(features, p=self.feat_drop, training=self.training)

        if self.use_relational_gnn:
            hetero_graph = self.hetero_graph
            graph_device = getattr(hetero_graph, 'device', None)
            if graph_device is not None and graph_device != features.device:
                hetero_graph = hetero_graph.to(features.device)
                self.hetero_graph = hetero_graph

            node_dict = {self.primary_ntype: features}
            output = features
            for layer_idx, conv in enumerate(self.rel_layers):
                node_dict = conv(hetero_graph, node_dict)
                output = node_dict[self.primary_ntype]
                if self.rel_norms is not None:
                    output = self.rel_norms[layer_idx](output)
                if layer_idx != len(self.rel_layers) - 1 and self.use_relu:
                    output = F.relu(output)
                output = F.dropout(output, p=self.feat_drop, training=self.training)
                node_dict = {self.primary_ntype: output}

            return output

        X_gnn = self.gnn_encoder(features, adj_t)

        return X_gnn

