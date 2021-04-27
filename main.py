
from options import args_parser
import random
import numpy as np
import torch
import csv
import sys
from dataloader import load_lookups, prepare_instance, prepare_instance_bert, MyDataset, my_collate, my_collate_bert,   load_lookups_MTL, prepare_instance_MTL, prepare_instance_bert_MTL, MyDataset, my_collate_MTL, my_collate_bert_MTL
from utils import early_stop, save_everything, save_everything_MTL
from models import pick_model
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import time
from train_test import train, test, train_MTL, test_MTL
from transformers import AdamW, get_linear_schedule_with_warmup


if __name__ == "__main__":
    args = args_parser()
    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    print(args)
    
    if sys.platform == 'win32':
        csv.field_size_limit(2147483647)
    else:    
        csv.field_size_limit(sys.maxsize)

    # load vocab and other lookups
    print("loading lookups...")
    desc_embed = args.lmbda > 0
    if args.MTL == 'Yes':
        dicts = load_lookups_MTL(args, desc_embed=desc_embed)
    else:
        dicts = load_lookups(args, desc_embed=desc_embed)

    model = pick_model(args, dicts)

    if not args.test_model:
        if not args.weight_tuning:
            if args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
            if args.optimizer == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, betas=(0.9, 0.999))
        else:
            optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=0.9)
    else:
        optimizer = None

    if args.model != 'elmo' and args.tune_wordemb == False:
        model.freeze_net()

    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    if args.MTL == 'Yes':
        metrics_hist_ccs = defaultdict(lambda: [])
        metrics_hist_te_ccs = defaultdict(lambda: [])
    else:
        pass

    if args.model.find("bert") != -1:
        if args.MTL == 'Yes':
            prepare_instance_func = prepare_instance_bert_MTL
        else:
            prepare_instance_func = prepare_instance_bert
    else:
        if args.MTL == 'Yes':
            prepare_instance_func = prepare_instance_MTL
        else:
            prepare_instance_func = prepare_instance

    train_instances = prepare_instance_func(dicts, args.data_path, args, args.MAX_LENGTH)
    print("train_instances {}".format(len(train_instances)))
    if args.version != 'mimic2':
        dev_instances = prepare_instance_func(dicts, args.data_path.replace('train','dev'), args, args.MAX_LENGTH)
        print("dev_instances {}".format(len(dev_instances)))
    else:
        dev_instances = None
    test_instances = prepare_instance_func(dicts, args.data_path.replace('train','test'), args, args.MAX_LENGTH)
    print("test_instances {}".format(len(test_instances)))

    if args.model.find("bert") != -1:
        if args.MTL == 'Yes':
            collate_func = my_collate_bert_MTL
        else:
            collate_func = my_collate_bert
    else:
        if args.MTL == 'Yes':
            collate_func = my_collate_MTL
        else:
            collate_func = my_collate

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
    if args.version != 'mimic2':
        dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
    else:
        dev_loader = None
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)


    if not args.test_model and args.model.find("bert") != -1:
        # Layerwise decreasing learning rates for Bio_ClinicalBERT. 
        if args.use_lr_layer_decay:
            optimizer = AdamW(
                [{'params':model.bert.embeddings.parameters(), 'lr': args.lr * args.lr_layer_decay ** (len(model.bert.encoder.layer) + 2)}]
                +
                [{'params': module_list_item.parameters(), 'lr': args.lr * args.lr_layer_decay ** (len(model.bert.encoder.layer) + 1 - index), } 
                for index, module_list_item in enumerate(model.bert.encoder.layer)]
                +
                [{'params':model.bert.pooler.parameters(), 'lr': args.lr * args.lr_layer_decay ** 1}]
                +
                [{'params':model.final.parameters(), 'lr': args.lr * args.lr_layer_decay ** 0}]
            , correct_bias = False, weight_decay = args.weight_decay)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False, weight_decay=args.weight_decay)  # To reproduce BertAdam specific behavior set correct_bias=False

    # Linearly decaying lr-schedule with warmup
    if not args.test_model and args.use_lr_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.warm_up * len(train_loader) * args.n_epochs, (1-args.warm_up) * len(train_loader) * args.n_epochs)
    else:
        lr_scheduler = None

    test_only = args.test_model is not None

    


    for epoch in range(args.n_epochs):
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))

        if not test_only:
            if args.debug:
                for param_group in optimizer.param_groups:
                    print("Learning rate: % .10f" % (param_group['lr']))
            
            epoch_start = time.time()
            if args.MTL == 'Yes':
                losses = train_MTL(args, model, optimizer, epoch, args.gpu, train_loader, lr_scheduler)
            else:
                losses = train(args, model, optimizer, epoch, args.gpu, train_loader, lr_scheduler)
            loss = np.mean(losses)
            epoch_finish = time.time()
            print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
        else:
            loss = np.nan

        fold = 'test' if args.version == 'mimic2' else 'dev'
        dev_instances = test_instances if args.version == 'mimic2' else dev_instances
        dev_loader = test_loader if args.version == 'mimic2' else dev_loader
        if epoch == args.n_epochs - 1:
            print("last epoch: testing on dev and test sets")
            test_only = True

        # test on dev
        evaluation_start = time.time()
        if args.MTL == 'Yes':
            metrics, metrics_ccs = test_MTL(args, model, args.data_path, fold, args.gpu, dicts, dev_loader)
        else:
            metrics = test(args, model, args.data_path, fold, args.gpu, dicts, dev_loader)
        evaluation_finish = time.time()
        print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
        if test_only or epoch == args.n_epochs - 1:
            if args.MTL == 'Yes':
                metrics_te, metrics_te_ccs = test_MTL(args, model, args.data_path, "test", args.gpu, dicts, test_loader)
            else:
                metrics_te = test(args, model, args.data_path, "test", args.gpu, dicts, test_loader)
        else:
            metrics_te = defaultdict(float)
            if args.MTL == 'Yes':
                metrics_te_ccs = defaultdict(float)
            else:
                pass
        metrics_tr = {'loss': loss}
        if args.MTL == 'Yes':
            metrics_all = (metrics, metrics_te, metrics_ccs, metrics_te_ccs, metrics_tr)
        else:
            metrics_all = (metrics, metrics_te, metrics_tr)

        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        if args.MTL == 'Yes':
            for name in metrics_all[2].keys():
                metrics_hist_ccs[name].append(metrics_all[2][name])
            for name in metrics_all[3].keys():
                metrics_hist_te_ccs[name].append(metrics_all[3][name])
            for name in metrics_all[4].keys():
                metrics_hist_tr[name].append(metrics_all[4][name])
            metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_ccs, metrics_hist_te_ccs, metrics_hist_tr)
            save_everything_MTL(args, metrics_hist_all, model, model_dir, None, args.criterion, test_only)
            sys.stdout.flush()
        else:
            for name in metrics_all[2].keys():
                metrics_hist_tr[name].append(metrics_all[2][name])
            metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)
            save_everything(args, metrics_hist_all, model, model_dir, None, args.criterion, test_only)
            sys.stdout.flush()

        if test_only:
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = pick_model(args, dicts)
