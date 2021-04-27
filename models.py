
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from gensim.models import KeyedVectors
from math import floor
import numpy as np
from transformers import AutoModel
from torch.autograd import Variable
import os

from elmo.elmo import Elmo
import json
from embeddings import build_pretrain_embedding, load_embeddings
from math import floor


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'], True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.use_elmo = args.use_elmo
        if self.use_elmo:
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
                             dropout=args.elmo_dropout, gamma=args.elmo_gamma)
            with open(args.elmo_options_file, 'r') as fin:
                _options = json.load(fin)
            self.feature_size += _options['lstm']['projection_dim'] * 2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]}

    def forward(self, x, target, text_inputs):
        features = [self.embed(x)]
        if self.use_elmo:
            elmo_outputs = self.elmo(text_inputs)
            elmo_outputs = elmo_outputs['elmo_representations'][0]
            features.append(elmo_outputs)
        x = torch.cat(features, dim=2)
        x = self.embed_drop(x)
        return x

class RAM(nn.Module):
    def __init__(self, kernel_size, stride, input_size, args):
        super(RAM, self).__init__()
        chan1=input_size
        chan2=int(chan1/2)
        chan3=int(chan2/2)
        self.down1 = nn.Sequential(
            nn.Conv1d(chan1, chan2, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan2),
            nn.Tanh(),
            nn.Conv1d(chan2, chan2, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan2)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(chan2, chan3, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan3),
            nn.Tanh(),
            nn.Conv1d(chan3, chan3, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan3)
        )
        self.lateral = nn.Sequential(
            nn.Conv1d(chan3, chan3, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan3),
            nn.Tanh(),
            nn.Conv1d(chan3, chan3, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan3)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(chan3, chan2, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan2),
            nn.Tanh(),
            nn.ConvTranspose1d(chan2, chan2, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan2)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(chan2, chan1, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan1),
            nn.Tanh(),
            nn.ConvTranspose1d(chan1, chan1, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(chan1)
        )
        self.dropout = nn.Dropout(p=args.dropout)
    def forward(self, x):
        D1 = self.down1(x)
        D2 = self.down2(D1)
        L = self.lateral(D2)
        U1 = self.up1(L)+D1
        U2 = self.up2(U1)*x
        U2 = torch.tanh(U2)
        res = self.dropout(U2)
        return res

class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()
        self.args = args
        self.Y = Y
        self.U = nn.Linear(input_size, Y)
        xavier_uniform_(self.U.weight)
        self.final = nn.Linear(input_size, Y)
        xavier_uniform_(self.final.weight)
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, text_inputs):
        att = self.U.weight.matmul(x.transpose(1, 2)) # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x) # [bs, Y, dim]
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        loss = self.loss_function(y, target)
        return y, loss

class OutputLayer_MTL(nn.Module):
    def __init__(self, args, Y, Y_ccs, dicts, input_size):
        super(OutputLayer_MTL, self).__init__()
        self.args = args
        self.Y = Y
        self.Y_ccs = Y_ccs
        self.U = nn.Linear(input_size, Y)
        self.U_ccs = nn.Linear(input_size, Y_ccs)
        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.U_ccs.weight)
        self.final = nn.Linear(input_size, Y)
        self.final_ccs = nn.Linear(input_size, Y_ccs)
        xavier_uniform_(self.final.weight)
        xavier_uniform_(self.final_ccs.weight)
        self.loss_function = nn.BCEWithLogitsLoss()
        if args.RAM:
            self.msm = RAM(3, 1, input_size, args)

    def forward(self, x, target, target_ccs, text_inputs):
        try:
            x = self.msm(x.transpose(1, 2)).transpose(1, 2)
        except AttributeError:
            x = x
        att = self.U.weight.matmul(x.transpose(1, 2)) # [bs, Y, seq_len]
        att_ccs = self.U_ccs.weight.matmul(x.transpose(1, 2))
        alpha = F.softmax(att, dim=2)
        alpha_ccs = F.softmax(att_ccs, dim=2)
        m = alpha.matmul(x) # [bs, Y, dim]
        m_ccs = alpha_ccs.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        y_ccs = self.final_ccs.weight.mul(m_ccs).sum(dim=2).add(self.final_ccs.bias)
        loss = self.loss_function(y, target)
        loss_ccs = self.loss_function(y_ccs, target_ccs)
        return y, y_ccs, loss, loss_ccs

class ELMo(nn.Module):
    def __init__(self, args, Y, dicts):
        super(ELMo, self).__init__()

        self.gpu = args.gpu

        # if args.embed_file:
        #     print("loading pretrained embeddings from {}".format(args.embed_file))
        #     if args.use_ext_emb:
        #         pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'], True)
        #         W = torch.from_numpy(pretrain_word_embedding)
        #     else:
        #         W = torch.Tensor(load_embeddings(args.embed_file))

        #     self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        #     self.embed.weight.data = W.clone()
        # else:
        #     # add 2 to include UNK and PAD
        #     self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        # self.feature_size = self.embed.embedding_dim

        self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
                            dropout=args.elmo_dropout, gamma=args.elmo_gamma)
        with open(args.elmo_options_file, 'r') as fin:
            _options = json.load(fin)
        self.feature_size = _options['lstm']['projection_dim'] * 2
        self.dropout = nn.Dropout(p=args.dropout)
        self.output_layer = OutputLayer(args, Y, dicts, self.feature_size)
        

    def forward(self, x, target, text_inputs):
        elmo_outputs = self.elmo(text_inputs)
        features = self.dropout((elmo_outputs)['elmo_representations'][0])
        y, loss = self.output_layer(features, target, text_inputs)
        return y, loss


class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()
        
        self.args = args
        self.word_rep = WordRep(args, Y, dicts)
        filter_size = int(args.filter_size)
        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform_(self.conv.weight)
        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x = torch.tanh(self.conv(x).transpose(1, 2))
        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class DR_CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(DR_CNN, self).__init__()
        
        self.args = args
        self.word_rep = WordRep(args, Y, dicts)
        filter_size = int(args.filter_size)
        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform_(self.conv.weight)
        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)
        self.lmbda = args.lmbda
        self.gpu = args.gpu
        if args.lmbda > 0:
            W = self.word_rep.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()
            kernel_size = 10
            self.label_conv = nn.Conv1d(100, args.num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
            xavier_uniform_(self.label_conv.weight)

            self.label_fc1 = nn.Linear(args.num_filter_maps, args.num_filter_maps)
            xavier_uniform_(self.label_fc1.weight)

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.output_layer.final.weight[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs

    def embed_descriptions(self, desc_data, gpu):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch
        
    def forward(self, x, target, text_inputs, desc_data):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x = torch.tanh(self.conv(x).transpose(1, 2))
        y, loss = self.output_layer(x, target, text_inputs)
        #run descriptions through description module
        b_batch = self.embed_descriptions(desc_data, True)
        #get l2 similarity loss
        diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class CNN_MTL(nn.Module):
    def __init__(self, args, Y, Y_ccs, dicts):
        super(CNN_MTL, self).__init__()

        self.args = args
        self.word_rep = WordRep(args, Y, dicts)
        filter_size = int(args.filter_size)
        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform_(self.conv.weight)
        self.output_layer = OutputLayer_MTL(args, Y, Y_ccs, dicts, args.num_filter_maps)

    def forward(self, x, target1, target2, text_inputs):
        x = self.word_rep(x, target1, text_inputs)
        x = x.transpose(1, 2)
        x = torch.tanh(self.conv(x).transpose(1, 2))
        y1, y2, loss1, loss2 = self.output_layer(x, target1, target2, text_inputs)
        return y1, y2, loss1, loss2

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class LogisticRegression(nn.Module):
    def __init__(self, args, Y, dicts):
        super(LogisticRegression, self).__init__()

        self.args = args
        self.word_rep = WordRep(args, Y, dicts)
        self.linear = nn.Linear(self.word_rep.feature_size, Y)
        self.loss_function = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target, text_inputs):
        if self.args.debug: print('x', x.shape)
        if self.args.debug: print('text_inputs', text_inputs.shape)
        y = self.word_rep(x, target, text_inputs)
        if self.args.debug: print('word_rep: ', y.shape)
        y = y.transpose(1,2) # [batch, embed_len, seq_len]
        y = y.sum(2) # From word embeddings to document embedding
        if self.args.debug: print('document_embedding: ', y.shape)

        y = self.linear(y)
        if self.args.debug: print('linear:  ', y.shape)
                
        if self.args.debug: print('target:  ', target.shape)
        loss = self.loss_function(y, target)

        return y, loss
    
    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class BERT(nn.Module):

    def __init__(self, args, Y):
        super(BERT, self).__init__()
        self.apply(self.init_bert_weights)

        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.final = nn.Linear(768, Y)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        # x: [bs, seq_len]
        hidden_states, attentions  = self.bert(input_ids, token_type_ids, attention_mask) # hidden_states: [bs, seq_len, 768]
        final_features = hidden_states[:,0,:]   # the 0-th hidden state is used for classification
        y = self.final(final_features)

        loss = F.binary_cross_entropy_with_logits(y, target)
        return y, loss

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def freeze_net(self):
        pass

class RNNmodel(nn.Module):
    def __init__(self, args, Y, dicts):
        super(RNNmodel, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        self.encoder = nn.GRU(100, args.nhid, batch_first=True, bidirectional=args.bidirectional)
        if args.bidirectional:
            self.output_layer = OutputLayer(args, Y, dicts, args.nhid*2)
        else:
            self.output_layer = OutputLayer(args, Y, dicts, args.nhid)

    def forward(self, x, target, text_inputs, mask=None):
        x = self.word_rep(x, target, text_inputs) # emb: [bs, len, 100]
        x, hn = self.encoder(x)
        logits, loss = self.output_layer(x, target, None)
        return logits, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class RNNmodel_MTL(nn.Module):
    def __init__(self, args, Y, Y_ccs, dicts):
        super(RNNmodel_MTL, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        self.encoder = nn.GRU(100, args.nhid, batch_first=True, bidirectional=args.bidirectional)
        if args.bidirectional:
            self.output_layer = OutputLayer_MTL(args, Y, Y_ccs, dicts, args.nhid*2)
        else:
            self.output_layer = OutputLayer_MTL(args, Y, Y_ccs, dicts, args.nhid)

    def forward(self, x, target1, target2, text_inputs, mask=None):
        x = self.word_rep(x, target1, text_inputs) # emb: [bs, len, 100]
        x, hn = self.encoder(x)
        logits1, logits2, loss1, loss2 = self.output_layer(x, target1, target2, None)
        return logits1, logits2, loss1, loss2

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform_(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN_MTL(nn.Module):

    def __init__(self, args, Y, Y_ccs, dicts):
        super(MultiResCNN_MTL, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform_(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer_MTL(args, Y, Y_ccs, dicts, self.filter_num * args.num_filter_maps)


    def forward(self, x, target1, target2, text_inputs):

        x = self.word_rep(x, target1, text_inputs)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y1, y2, loss1, loss2 = self.output_layer(x, target1, target2, text_inputs)

        return y1, y2, loss1, loss2

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


def pick_model(args, dicts):
    if args.MTL == 'Yes':
        Y = len(dicts['ind2c'])
        Y_ccs = len(dicts['ind2c_ccs'])
        if args.model == 'caml':
            model = CNN_MTL(args, Y, Y_ccs, dicts)
        elif args.model == 'elmo':
            model = ELMo(args, Y, Y_ccs, dicts)
        elif args.model == 'bert':
            model = BERT(args, Y)
        elif args.model == 'logistic_regression':
            model = LogisticRegression(args, Y, Y_ccs, dicts)
        elif args.model == 'MultiResCNN':
            model = MultiResCNN_MTL(args, Y, Y_ccs, dicts)
        elif args.model == 'GRU':
            model = RNNmodel_MTL(args, Y, Y_ccs, dicts)
        else:
            raise RuntimeError("wrong model name")
    else:
        Y = len(dicts['ind2c'])
        if args.model == 'caml':
            model = CNN(args, Y, dicts)
        elif args.model == 'elmo':
            model = ELMo(args, Y, dicts)
        elif args.model == 'bert':
            model = BERT(args, Y)
        elif args.model == 'logistic_regression':
            model = LogisticRegression(args, Y, dicts)
        elif args.model == 'MultiResCNN':
            model = MultiResCNN(args, Y, dicts)
        elif args.model == 'GRU':
            model = RNNmodel(args, Y, dicts)
        elif args.model == 'DR_CAML':
            model = DR_CNN(args, Y, dicts)
        else:
            raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
