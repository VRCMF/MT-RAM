import torch
import numpy as np
from utils import all_metrics, print_metrics


max_grad_norm = 1.0

def train(args, model, optimizer, epoch, gpu, data_loader, lr_scheduler = None):
    print("EPOCH %d" % epoch)
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    losses = []
    model.train()
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        if args.model.find("bert") != -1:
            inputs_id, segments, masks, labels = next(data_iter)
            inputs_id = torch.LongTensor(inputs_id).to(device)
            segments = torch.LongTensor(segments).to(device)
            masks = torch.LongTensor(masks).to(device)
            labels = torch.FloatTensor(labels).to(device)
            output, loss = model(inputs_id, segments, masks, labels)
        else:
            inputs_id, labels, text_inputs, inputs_mask, descs = next(data_iter)
            inputs_id, labels = torch.LongTensor(inputs_id).to(device, non_blocking=False), torch.FloatTensor(labels).to(device, non_blocking=False)
            text_inputs, inputs_mask = text_inputs.to(device, non_blocking=False), torch.LongTensor(inputs_mask).to(device, non_blocking=False)
            desc_embed = args.lmbda > 0
            if desc_embed:
                desc_data = descs
            else:
                desc_data = None
            output, loss = model(inputs_id, labels, text_inputs, desc_data)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        losses.append(loss.item())
        if i % args.print_every == 0:
            print("Train epoch: {:>2d} [batch #{:>4d}, max_seq_len {:>4d}]\tLoss: {:.6f}".format(epoch, i, inputs_id.size()[1], loss.item()))
            if args.debug:
                for param_group in optimizer.param_groups:
                    print("Learning rate: % .10f" % (param_group['lr']))
    return losses

def train_MTL(args, model, optimizer, epoch, gpu, data_loader, lr_scheduler = None):
    print("EPOCH %d" % epoch)
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    losses = []
    model.train()
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        if args.model.find("bert") != -1:
            inputs_id, segments, masks, labels = next(data_iter)
            inputs_id = torch.LongTensor(inputs_id).to(device)
            segments = torch.LongTensor(segments).to(device)
            masks = torch.LongTensor(masks).to(device)
            labels = torch.FloatTensor(labels).to(device)
            output, loss = model(inputs_id, segments, masks, labels)
        else:
            inputs_id, labels, labels_ccs, text_inputs, inputs_mask = next(data_iter)
            #transform inputs_id, labels, text_inputs, inputs_mask  to appropriate class
            inputs_id = torch.LongTensor(inputs_id).to(device, non_blocking=False)
            labels = torch.FloatTensor(labels).to(device, non_blocking=False)
            labels_ccs = torch.FloatTensor(labels_ccs).to(device, non_blocking=False)
            text_inputs, inputs_mask = text_inputs.to(device, non_blocking=False), torch.LongTensor(inputs_mask).to(device, non_blocking=False)
            # careful about the parameters (change them)
            output1, output2, loss1, loss2 = model(inputs_id, labels, labels_ccs, text_inputs)
        optimizer.zero_grad()
        #loss_weight_ICD, loss_weight_CCS = args.loss_weight_ICD, args.loss_weight_CCS
        loss = (1-args.loss_weight_CCS)*loss1 + args.loss_weight_CCS*loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        losses.append(loss.item())
        if i % args.print_every == 0:
            print("Train epoch: {:>2d} [batch #{:>4d}, max_seq_len {:>4d}]\tLoss: {:.6f}".format(epoch, i, inputs_id.size()[1], loss.item()))
            if args.debug:
                for param_group in optimizer.param_groups:
                    print("Learning rate: % .10f" % (param_group['lr']))
    return losses

def test_MTL(args, model, data_path, fold, gpu, dicts, data_loader):
    filename = data_path.replace('train', fold)
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])
    y, yhat, yhat_raw, y_ccs, yhat_ccs, yhat_raw_ccs, hids, losses, losses_ccs = [], [], [], [], [], [], [], [], []

    model.eval()
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():
            if args.model.find("bert") != -1:
                inputs_id, segments, masks, labels = next(data_iter)
                inputs_id = torch.LongTensor(inputs_id).to(device)
                segments = torch.LongTensor(segments).to(device)
                masks = torch.LongTensor(masks).to(device)
                labels = torch.FloatTensor(labels).to(device)
                output, loss = model(inputs_id, segments, masks, labels)
            else:
                inputs_id, labels, labels_ccs, text_inputs, inputs_mask = next(data_iter)
                #transform inputs_id, labels, text_inputs, inputs_mask  to appropriate class
                inputs_id = torch.LongTensor(inputs_id).to(device, non_blocking=False)
                labels = torch.FloatTensor(labels).to(device, non_blocking=False)
                labels_ccs = torch.FloatTensor(labels_ccs).to(device, non_blocking=False)
                text_inputs, inputs_mask = text_inputs.to(device, non_blocking=False), torch.LongTensor(inputs_mask).to(device, non_blocking=False)
                # careful about the parameters (change them)
                output1, output2, loss1, loss2 = model(inputs_id, labels, labels_ccs, text_inputs)
            output1 = torch.sigmoid(output1) # Output to probabilities with sigmoid
            output2 = torch.sigmoid(output2)
            output1 = output1.data.cpu().numpy()
            output2 = output2.data.cpu().numpy()
            losses.append(loss1.item())
            losses_ccs.append(loss2.item())
            target_data = labels.data.cpu().numpy()
            target_data_ccs = labels_ccs.data.cpu().numpy()
            yhat_raw.append(output1)
            yhat_raw_ccs.append(output2)
            output1 = np.round(output1)
            output2 = np.round(output2)
            y.append(target_data)
            y_ccs.append(target_data_ccs)
            yhat.append(output1)
            yhat_ccs.append(output2)

    y = np.concatenate(y, axis=0)
    y_ccs = np.concatenate(y_ccs, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_ccs = np.concatenate(yhat_ccs, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)
    yhat_raw_ccs = np.concatenate(yhat_raw_ccs, axis=0)

    k = 5 if num_labels == 50 else [8,15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    metrics_ccs = all_metrics(yhat_ccs, y_ccs, k=k, yhat_raw=yhat_raw_ccs)
    print_metrics(metrics)
    print_metrics(metrics_ccs)
    metrics['loss_%s' % fold] = np.mean(losses)
    metrics_ccs['loss_%s' % fold] = np.mean(losses_ccs)
    return metrics, metrics_ccs

def test(args, model, data_path, fold, gpu, dicts, data_loader):
    filename = data_path.replace('train', fold)
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])
    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():
            if args.model.find("bert") != -1:
                inputs_id, segments, masks, labels = next(data_iter)
                inputs_id = torch.LongTensor(inputs_id).to(device)
                segments = torch.LongTensor(segments).to(device)
                masks = torch.LongTensor(masks).to(device)
                labels = torch.FloatTensor(labels).to(device)
                output, loss = model(inputs_id, segments, masks, labels)
            else:
                inputs_id, labels, text_inputs, inputs_mask, descs = next(data_iter)
                inputs_id, labels = torch.LongTensor(inputs_id).to(device, non_blocking=False), torch.FloatTensor(labels).to(device, non_blocking=False)
                text_inputs, inputs_mask = text_inputs.to(device, non_blocking=False), torch.LongTensor(inputs_mask).to(device, non_blocking=False)
                desc_embed = args.lmbda > 0
                if desc_embed:
                    desc_data = descs
                else:
                    desc_data = None
                output, loss = model(inputs_id, labels, text_inputs, desc_data)
            output = torch.sigmoid(output) # Output to probabilities with sigmoid
            output = output.data.cpu().numpy()
            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()
            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    k = 5
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics
