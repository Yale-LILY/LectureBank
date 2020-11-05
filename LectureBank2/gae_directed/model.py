import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import GraphConvolution, RelationalGraphConvolution


class GCNModelVAE_Semi(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, class_dim):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_feat_dim: (int): write your description
            hidden_dim1: (int): write your description
            hidden_dim2: (int): write your description
            dropout: (str): write your description
            class_dim: (int): write your description
        """
        super(GCNModelVAE_Semi, self).__init__()
        # self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        # self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.gc1 = RelationalGraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = RelationalGraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = RelationalGraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)


        self.dc = InnerProductDecoder(dropout, act=lambda x: x)


        'add another layer to predict node labels'
        self.W_node = Parameter(torch.Tensor(hidden_dim2, class_dim))
        torch.nn.init.xavier_uniform_(self.W_node)


    def encode(self, x, adj):
        """
        Encode x as a binary representation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            adj: (todo): write your description
        """
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        """
        Reparameters

        Args:
            self: (todo): write your description
            mu: (todo): write your description
            logvar: (bool): write your description
        """
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):

        # import pdb;pdb.set_trace()
        'adj is a list of adj vars'
        'mu = hidden layer, logvar = last layer (shape: torch.Size([2039, 16]))'
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)

        'return the node predictions'
        pred_nodes = torch.mm(logvar,self.W_node) # shape torch.Size([2039, 322])

        return self.dc(z), mu, logvar, pred_nodes

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_feat_dim: (int): write your description
            hidden_dim1: (int): write your description
            hidden_dim2: (int): write your description
            dropout: (str): write your description
        """
        super(GCNModelVAE, self).__init__()
        # self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        # self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc1 = RelationalGraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = RelationalGraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = RelationalGraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        """
        Encode x as a binary representation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            adj: (todo): write your description
        """
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        """
        Reparameters

        Args:
            self: (todo): write your description
            mu: (todo): write your description
            logvar: (bool): write your description
        """
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        'adj is a list of adj vars'
        'mu = hidden layer, logvar = last layer'

        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        """
        Initialize the chat initialization.

        Args:
            self: (todo): write your description
            dropout: (str): write your description
            act: (str): write your description
            torch: (todo): write your description
            sigmoid: (str): write your description
        """
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            z: (todo): write your description
        """
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
