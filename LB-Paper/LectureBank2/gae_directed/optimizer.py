import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function_label(preds, labels, mu, logvar, n_nodes, norm, pos_weight, pred_nodes,tags_nodes):

    '''
    Added pred_nodes and tags_nodes for prediction loss
    :param preds:
    :param labels:
    :param mu:
    :param logvar:
    :param n_nodes:
    :param norm:
    :param pos_weight:
    :param pred_nodes:
    :param tags_nodes:
    :return:
    '''
    # convert to Tensor
    'what is pos_weight for?'
    pos_weight_tensor = torch.tensor(pos_weight)

    mse_loss = torch.nn.MSELoss()
    class_loss =torch.nn.CrossEntropyLoss()
    class_cost = class_loss(pred_nodes,tags_nodes)

    # combine two label matrices
    combined_labels = torch.add(labels[0],labels[1]) - torch.eye(labels[0].shape[0])
    cost = mse_loss(preds, combined_labels)


    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    # print ('Cost ', cost)
    # print ('KLD ', KLD)

    return cost + KLD + class_cost

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):

    '''
    Added pred_nodes and tags_nodes for prediction loss
    :param preds:
    :param labels:
    :param mu:
    :param logvar:
    :param n_nodes:
    :param norm:
    :param pos_weight:
    :param pred_nodes:
    :param tags_nodes:
    :return:
    '''
    # convert to Tensor
    'what is pos_weight for?'
    pos_weight_tensor = torch.tensor(pos_weight)

    mse_loss = torch.nn.MSELoss()

    # import pdb;
    # pdb.set_trace()

    # combine two label matrices
    combined_labels = torch.add(labels[0],labels[1]) - torch.eye(labels[0].shape[0])
    cost = mse_loss(preds, combined_labels)


    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))


    return cost + KLD


def loss_function_relation_semi(preds, labels, mu, logvar, n_nodes, norm, pos_weight, pred_nodes,tags_nodes):

    '''
    Added pred_nodes and tags_nodes for prediction loss
    :param preds:
    :param labels:
    :param mu:
    :param logvar:
    :param n_nodes:
    :param norm:
    :param pos_weight:
    :param pred_nodes:
    :param tags_nodes:
    :return:
    '''
    # convert to Tensor
    class_loss =torch.nn.CrossEntropyLoss()
    class_cost = class_loss(pred_nodes,tags_nodes)

    pos_weight_tensor = torch.tensor(pos_weight[0])
    cost_0 = norm[0] * F.binary_cross_entropy_with_logits(preds, labels[0], pos_weight=pos_weight_tensor)

    pos_weight_tensor = torch.tensor(pos_weight[1])
    cost_1 = norm[1] * F.binary_cross_entropy_with_logits(preds, labels[1], pos_weight=pos_weight_tensor)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost_0 + cost_1 + KLD + class_cost

def loss_function_relation(preds, labels, mu, logvar, n_nodes, norm, pos_weight):

    # combined_labels = torch.add(labels[0], labels[1]) - torch.eye(labels[0].shape[0])

    # import pdb;pdb.set_trace()
    # convert to Tensor
    pos_weight_tensor = torch.tensor(pos_weight[0])
    cost_0 = norm[0] * F.binary_cross_entropy_with_logits(preds, labels[0], pos_weight=pos_weight_tensor)

    pos_weight_tensor = torch.tensor(pos_weight[1])
    cost_1 = norm[1] * F.binary_cross_entropy_with_logits(preds, labels[1], pos_weight=pos_weight_tensor)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))


    return cost_0 + cost_1 + KLD

def loss_function_original(preds, labels, mu, logvar, n_nodes, norm, pos_weight):

    combined_labels = torch.add(labels[0], labels[1]) - torch.eye(labels[0].shape[0])

    # import pdb;pdb.set_trace()
    # convert to Tensor
    pos_weight_tensor = torch.tensor(pos_weight)
    cost = norm * F.binary_cross_entropy_with_logits(preds, combined_labels, pos_weight=pos_weight_tensor)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost + KLD