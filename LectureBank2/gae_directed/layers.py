import torch

import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_features: (int): write your description
            out_features: (int): write your description
            dropout: (str): write your description
            act: (str): write your description
            F: (int): write your description
            relu: (todo): write your description
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters.

        Args:
            self: (todo): write your description
        """
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            adj: (todo): write your description
        """
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RelationalGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        """
        Initialize features.

        Args:
            self: (todo): write your description
            in_features: (int): write your description
            out_features: (int): write your description
            dropout: (str): write your description
            act: (str): write your description
            F: (int): write your description
            relu: (todo): write your description
        """
        super(RelationalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        'weight is for all (W0 in the paper; other two are separate weights'
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_dc = Parameter(torch.Tensor(in_features, out_features))
        self.weight_dd = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    # def __init__(self, in_features, out_features, bias=False):
    #     super(RelationalGraphConvolution, self).__init__()
    #     self.in_features = in_features
    #     self.out_features = out_features
    #     self.weight = Parameter(torch.Tensor(in_features, out_features)) # this is for all
    #     self.weight_dc = Parameter(torch.Tensor(in_features, out_features))
    #     self.weight_dd = Parameter(torch.Tensor(in_features, out_features))
    #
    #     if bias:
    #         self.bias = Parameter(torch.Tensor(out_features))
    #     else:
    #         self.register_parameter('bias', None)
    #     self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the weights.

        Args:
            self: (todo): write your description
        """
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight_dc)
        torch.nn.init.xavier_uniform_(self.weight_dd)

    def forward(self, input, adj):
        '''
        TODO:

        :param x: x is a list of features: whole, dd and dc (same shape)
        :param adj: adj is a list of features: whole, dd and dc (same shape)
        :return:
        '''
        'adj will be a list of adj, list of 2'
        input = F.dropout(input, self.dropout, self.training)

        'all_adj is all of the adj'
        all_adj = adj[0].add(adj[1]).sub(torch.eye(adj[0].shape[0]).to_sparse())

        # for over-all
        support = torch.mm(input, self.weight)
        output = torch.spmm(all_adj, support)

        # for dc
        support_dc = torch.mm(input, self.weight_dc)
        output_dc = torch.spmm(adj[0], support_dc)

        # for dd
        support_dd = torch.mm(input, self.weight_dd)
        output_dd = torch.spmm(adj[1], support_dd)

        # import pdb;pdb.set_trace()
        # add all
        # final_output = (output + output_dc + output_dd)/3
        # final_output = torch.add(output,torch.div(output_dc + output_dd,2))
        # final_output = ((output_dc + output_dd)/2 + output)/2


        # works for tf-idf
        final_output = (output + output_dc + output_dd)/3

        return final_output

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
