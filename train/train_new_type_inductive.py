# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
# 将 data 文件夹的路径添加到 sys.path 中
sys.path.append(data_folder)

from ddi_dataset_ind import DDIDataset, BatchLoader

from torch.utils.data import DataLoader

def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    
    y_pred_label = (y_pred >= 0.5).astype(np.int32)
    
    acc = metrics.accuracy_score(y_true, y_pred_label)
    auc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred_label, zero_division=0)
    
    p = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    r = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    ap = metrics.average_precision_score(y_true, y_pred)
    
    return acc, auc, f1, p, r, ap

def get_drop_rate_stats(drop_rate_list):
    drop_rate_stats = {
        "max" : 0.0,
        "min" : 0.0,
        "mean" : 0.0
    }
    
    if len(drop_rate_list) == 0:
        return drop_rate_stats
    
    drop_rate_stats["max"] = max(drop_rate_list)
    drop_rate_stats["min"] = min(drop_rate_list)
    drop_rate_stats["mean"] = sum(drop_rate_list) / len(drop_rate_list)

    return drop_rate_stats

@torch.no_grad()
def evaluate(model, args, loader, set_len, replace_graph):
    if args.graph == 'train':
       model.knowledge_graph = replace_graph.to(args.device) 
    cur_num = 0
    y_pred_all, y_true_all = [], []
    
    for batch in loader:
        graph_batch_1, graph_batch_2, \
        graph_batch_old_1, graph_batch_old_2, \
        ddi_type, y_true = batch
        
        y_pred = model.forward(
            graph_batch_1, graph_batch_2,
            graph_batch_old_1, graph_batch_old_2, ddi_type
        )
        
        y_pred_all.append(y_pred.detach().cpu())
        y_true_all.append(torch.LongTensor(y_true))
        
        cur_num += graph_batch_1.num_graphs // 2
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
    
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    
    return calc_metrics(y_pred, y_true)

def train(model, args, summary_writer):
    train_set = DDIDataset(args.dataset, "train")
    valid_set = DDIDataset(args.dataset, "valid")
    either_set = DDIDataset(args.dataset, "either")
    both_set = DDIDataset(args.dataset, "both")
    
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    either_set_len = len(either_set)
    both_set_len = len(both_set)
    
    full_knowledge_graph = torch.load(f"../dataset/data/{args.dataset}/inductive/all_graph.pt")
    train_knowledge_graph = torch.load(f"../dataset/data/{args.dataset}/inductive/train_graph.pt")
    
    batch_loader = BatchLoader(args)
    
    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    either_loader = DataLoader(
        either_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    both_loader = DataLoader(
        both_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.dataset == 'drugbank':
        target_epoch = 200
    else:
        target_epoch = 50
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < target_epoch else 0.1),
        last_epoch=args.start_epoch-1
    )
    
    max_valid_acc, max_either_acc, max_both_acc = 0.0, 0.0, 0.0
    
    best_valid_epoch, best_either_epoch, best_both_epoch = 0, 0, 0
    
    global_step = 0
    
    for epoch in range(args.num_epoch):
        print(f"Epoch: {args.start_epoch + epoch}")
        if args.graph == 'train':

            model.knowledge_graph = train_knowledge_graph.to(args.device)
        train_loss = 0.0
        cur_num = 0
        y_pred_all, y_true_all = [], []
        train_set.do_shuffle()
        model.BGNN.drop_rate_list.clear()
        
        model.train()
        for i, batch in enumerate(train_loader):
            graph_batch_1, graph_batch_2, _, _, ddi_type, y_true = batch
            y_true = torch.Tensor(y_true).to(args.device)
            
            y_pred = model.forward(
                graph_batch_1, graph_batch_2,
                None, None, ddi_type
            )
            loss = criterion(y_pred, y_true)
            train_loss += loss.item()
            
            y_pred_all.append(y_pred.detach().sigmoid().cpu())
            y_true_all.append(y_true.detach().long().cpu())
            
            dr_stats = get_drop_rate_stats(model.BGNN.drop_rate_list)
            dr_stats_print = [f"{val:.4f}" for val in dr_stats.values()]
            dr_stats_print = ", ".join(dr_stats_print)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_num += graph_batch_1.num_graphs // 2
            sys.stdout.write(
                f"\r{cur_num} / {train_set_len}, "
                f"{(train_loss / (i + 1)):.6f}, "
                f"{dr_stats_print}"
                "          "
            )
            sys.stdout.flush()
        
        y_pred = torch.cat(y_pred_all)
        y_true = torch.cat(y_true_all)
        train_acc, train_auc, train_f1, train_p, train_r, train_ap = \
            calc_metrics(y_pred, y_true)
        print()
        print(
            f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n"
            f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}"
        )
        
        summary_writer.add_scalar('train/ACC', train_acc, global_step)
        summary_writer.add_scalar('train/AUC', train_auc, global_step)
        summary_writer.add_scalar('train/F1', train_f1, global_step)
        summary_writer.add_scalar('train/P', train_p, global_step)
        summary_writer.add_scalar('train/R', train_r, global_step)
        summary_writer.add_scalar('train/AP', train_ap, global_step)
        
        model.eval()
        valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap = \
            evaluate(model, args, valid_loader, valid_set_len, full_knowledge_graph)
        print()
        print(
            f"Valid ACC:  {valid_acc:.4f}, Valid AUC:  {valid_auc:.4f}, Valid F1:  {valid_f1:.4f}\n"
            f"Valid P:    {valid_p:.4f}, Valid R:    {valid_r:.4f}, Valid AP:  {valid_ap:.4f}"
        )
        
        summary_writer.add_scalar('valid/ACC', valid_acc, global_step)
        summary_writer.add_scalar('valid/AUC', valid_auc, global_step)
        summary_writer.add_scalar('valid/F1', valid_f1, global_step)
        summary_writer.add_scalar('valid/P', valid_p, global_step)
        summary_writer.add_scalar('valid/R', valid_r, global_step)
        summary_writer.add_scalar('valid/AP', valid_ap, global_step)
        
        either_acc, either_auc, either_f1, either_p, either_r, either_ap = \
            evaluate(model, args, either_loader, either_set_len, full_knowledge_graph)
        print()
        print(
            f"Either ACC:  {either_acc:.4f}, Either AUC:  {either_auc:.4f}, Either F1:  {either_f1:.4f}\n"
            f"Either P:    {either_p:.4f}, Either R:    {either_r:.4f}, Either AP:  {either_ap:.4f}"
        )
        
        summary_writer.add_scalar('either/ACC', either_acc, global_step)
        summary_writer.add_scalar('either/AUC', either_auc, global_step)
        summary_writer.add_scalar('either/F1', either_f1, global_step)
        summary_writer.add_scalar('either/P', either_p, global_step)
        summary_writer.add_scalar('either/R', either_r, global_step)
        summary_writer.add_scalar('either/AP', either_ap, global_step)
        
        both_acc, both_auc, both_f1, both_p, both_r, both_ap = \
            evaluate(model, args, both_loader, both_set_len, full_knowledge_graph)
        print()
        print(
            f"Both ACC:    {both_acc:.4f}, Both AUC:    {both_auc:.4f}, Both F1:    {both_f1:.4f}\n"
            f"Both P:      {both_p:.4f}, Both R:      {both_r:.4f}, Both AP:    {both_ap:.4f}"
        )
        
        summary_writer.add_scalar('both/ACC', both_acc, global_step)
        summary_writer.add_scalar('both/AUC', both_auc, global_step)
        summary_writer.add_scalar('both/F1', both_f1, global_step)
        summary_writer.add_scalar('both/P', both_p, global_step)
        summary_writer.add_scalar('both/R', both_r, global_step)
        summary_writer.add_scalar('both/AP', both_ap, global_step)
        
        global_step += 1
        
        path = f"./save/{args.dataset}_{args.kg_graph}_{args.mode}_bgnn{args.gnn_model}_tgnn{args.tgnn_model}_bs{args.batch_size}_lr{args.lr}"
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            best_valid_epoch = args.start_epoch + epoch
            torch.save(model.state_dict(), os.path.join(path, f"model_{args.start_epoch + epoch}.pt"))
            print(f"BEST VALID IN EPOCH {args.start_epoch + epoch}")
        
        if either_acc > max_either_acc:
            max_either_acc = either_acc
            best_either_epoch = args.start_epoch + epoch
            torch.save(model.state_dict(), os.path.join(path, f"model_{args.start_epoch + epoch}.pt"))
            print(f"BEST EITHER IN EPOCH {args.start_epoch + epoch}")
        
        if both_acc > max_both_acc:
            max_both_acc = both_acc
            best_both_epoch = args.start_epoch + epoch
            torch.save(model.state_dict(), os.path.join(path, f"model_{args.start_epoch + epoch}.pt"))
            print(f"BEST BOTH IN EPOCH {args.start_epoch + epoch}")
        
        scheduler.step()
        with open(os.path.join(path, f"inductive.txt"), 'a+') as f:
            f.write(f"**********************Training Epoch:{epoch}**************************\n")
            f.write(f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n")
            f.write(f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}\n")
            f.write(f"Valid ACC: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid F1: {valid_f1:.4f}\n")
            f.write(f"Valid P:   {valid_p:.4f}, Valid R:   {valid_r:.4f}, Valid AP: {valid_ap:.4f}\n")
            f.write(f"Either ACC:  {either_acc:.4f}, Either AUC:  {either_auc:.4f}, Either F1:  {either_f1:.4f}\n")
            f.write(f"Either P:    {either_p:.4f}, Either R:    {either_r:.4f}, Either AP:  {either_ap:.4f}\n")
            f.write(f"Both ACC:    {both_acc:.4f}, Both AUC:    {both_auc:.4f}, Both F1:    {both_f1:.4f}\n")
            f.write(f"Both P:      {both_p:.4f}, Both R:      {both_r:.4f}, Both AP:    {both_ap:.4f}\n")
            f.write(f"BEST VALID IN EPOCH {best_valid_epoch}\n")
            f.write(f"BEST EITHER IN EPOCH {best_either_epoch}\n")
            f.write(f"BEST BOTH IN EPOCH {best_both_epoch}\n")
        print()
        print()
