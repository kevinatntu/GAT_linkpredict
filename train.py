"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

'''
Requirements:
pip install dgl-cu101
pip install --pre dgl

Command:
python train.py --dataset=cora --num-layers=2 --epochs=[#epoch for upstream training] --num_sampling=[#run] --num_mlp_loop=[#epoch of perceptron/mlp]

default: epochs=100, num_mlp_loop=100

'''

import argparse
import numpy as np
import networkx as nx
import time
import os
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gat import GAT, LNet, MLP
from utils import EarlyStopping
import random
import json
import logging

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import scipy.stats

import pickle

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def accuracy(logits, labels):
    if len(logits.shape) == 1:
        indices = logits
    else:
        _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

def main(args):
    # load and preprocess dataset
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=args.logfile,
                    filemode='w')

    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
    
    # Generate seeds
    seed_lst = np.random.randint(low=1, high=9999, size=args.num_sampling * 2)

    row_0 = {
        'gat': [],
        "perceptron": [],
        "mlp": [],
        "lsvm": [],
        "rsvm" : [],
        "dt": [],
        "rf": [],
        "lgbm": []
    }

    fix = {
        'gat': [],
        "perceptron": [],
        "mlp": [],
        "lsvm": [],
        "rsvm" : [],
        "dt": [],
        "rf": [],
        "lgbm": []
    }

    no_fix = {
        'gat': [],
        "perceptron": [],
        "mlp": [],
        "lsvm": [],
        "rsvm" : [],
        "dt": [],
        "rf": [],
        "lgbm": []
    }

    if not os.path.exists('./cls'):
        os.mkdir('./cls')
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./results'):
        os.mkdir('./results')

    for run in range(args.num_sampling):
        # set seed
        args.first_seed = seed_lst[run]
        args.second_seed = seed_lst[run+1]

        print("Run {}: first seed={}, second seed={}".format(args.first_seed, args.second_seed))
        
        '''
        First model
        '''

        random.seed(args.first_seed)
        np.random.seed(args.first_seed)
        torch.manual_seed(args.first_seed)

        # Create graph
        g = data.graph
        # add self loop
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        n_edges = g.number_of_edges()
        # create model

        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT(g,
                    args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
        # check model
        if args.check_model:
            print(model)
            print(model.gat_layers[-1])
        
        if args.early_stop:
            stopper = EarlyStopping(patience=100)
        if cuda:
            model.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        '''
        cc = 0
        load_start_num = 6 # load row 0's cls
        name = []
        for param_tensor in model.state_dict():
            #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            cc += 1
            if cc > load_start_num: # 3: our goal, 6: remember nothing
                name.append(param_tensor)
        
        old_dict = {}
        
        #with open('./model_seed{}.pkl'.format(args.first_seed), 'rb') as f:
        #    pretrained_dict = pickle.load(f)
        pretrained_dict = torch.load('./model/model_seed{}.pkl'.format(args.first_seed))
            
        #print(pretrained_dict)

        w_count = 0
        #name = [gat_layers.1.attn_l, gat_layers.1.attn_r, gat_layers.1.fc.weight, gat_layers.2.attn_l, gat_layers.2.attn_r, gat_layers.2.fc.weight]
        for k, v in pretrained_dict.items():
            if k in name:
                print(k)
                old_dict[k] = v
                w_count += 1
            else:
                old_dict[k] = model.state_dict()[k]
                #model.load_state_dict(old_dict[k])

        print(w_count)
        
        if args.load_row0_cls:
            model.load_state_dict(old_dict)
            

            #for param in model.gat_layers[0].parameters():
            #    param.requires_grad = False
            #for param in model.gat_layers[1].parameters():
            #    param.requires_grad = False

            for param in model.gat_layers[2].parameters():
                param.requires_grad = False
        '''

        # use optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits, embedding = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[train_mask], labels[train_mask])

            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate(model, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model):   
                        break

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                format(epoch, np.mean(dur), loss.item(), train_acc,
                        val_acc, n_edges / np.mean(dur) / 1000))

        print()
        #if args.early_stop:
        #    model.load_state_dict(torch.load('es_checkpoint.pt'))
        #acc = evaluate(model, features, labels, test_mask)
        #print("Test Accuracy {:.4f}".format(acc))

        # save model
        torch.save(model.state_dict(), './model/model_seed{}.pkl'.format(args.first_seed))

        model.eval()

        # Train downstream cls
        print("\nFirst model, seed: ", args.first_seed)

        # Get embedding
        with torch.no_grad():
            logits, embedding = model(features)
            acc = accuracy(logits[test_mask], labels[test_mask])
            print("GAT cls Test Accuracy {:.4f}".format(acc))
            row_0['gat'].append(acc)

        # linear
        perceptron = LNet(embedding[train_mask].shape[1], n_classes)
        if cuda:
            perceptron = perceptron.cuda()
        perceptron.train()
        opt = torch.optim.Adam(perceptron.parameters(), lr=1e-3)

        for e in range(args.num_mlp_loop):
            perceptron_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(embedding[train_mask], labels[train_mask]),
                                                            batch_size=32,
                                                            shuffle=True,
                                                            num_workers=4)
            #pbar = tqdm(enumerate(mlp_dataset), total=len(mlp_dataset))
            pbar = enumerate(perceptron_dataset)
            n_correct = 0
            n_all = 0
            total_loss = 0
            for sidx, tokens in pbar:
                if cuda:
                    inputs, input_labels = tokens[0].cuda(), tokens[1].cuda()
                else:
                    inputs, input_labels = tokens[0], tokens[1]
                    
                logits = perceptron(inputs)
                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, n_classes), input_labels.view(-1))
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = input_labels.detach().cpu().numpy().flatten()

                n_correct += np.sum(pred_flat == labels_flat)
                n_all += len(labels_flat)
                train_scores = n_correct / n_all

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                #pbar.set_postfix({'loss': total_loss/(sidx+1), 'accu': train_scores})
        
        torch.save(perceptron.state_dict(), './cls/perceptron_{}.pkl'.format(args.first_seed))
        
        perceptron.eval()
        with torch.no_grad():
            if cuda:
                predict = perceptron(embedding[test_mask].cuda())
            else:
                predict = perceptron(embedding[test_mask])
            #predict = predict.cpu().detach().numpy()
            #predict = np.argmax(predict, axis=1).flatten()
            #scores = f1_score(pred_flat, labels[test_mask].flatten(), average='macro')
            #scores = calculate_scores(test_label.flatten(), predict, mode=args.task)
            acc = accuracy(predict, labels[test_mask])
            print("Perceptron training, acc: {}".format(acc))
            row_0['perceptron'].append(acc)

        # MLP
        mlp = MLP(embedding[train_mask].shape[1], embedding[train_mask].shape[1]*2, n_classes)
        if cuda:
            mlp = mlp.cuda()
        mlp.train()
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        for e in range(args.num_mlp_loop):
            mlp_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(embedding[train_mask], labels[train_mask]),
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=4)
            #pbar = tqdm(enumerate(mlp_dataset), total=len(mlp_dataset))
            pbar = enumerate(mlp_dataset)
            n_correct = 0
            n_all = 0
            total_loss = 0
            for sidx, tokens in pbar:
                if cuda:
                    inputs, input_labels = tokens[0].cuda(), tokens[1].cuda()
                else:
                    inputs, input_labels = tokens[0], tokens[1]

                logits = mlp(inputs)
                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, n_classes), input_labels.view(-1))
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = input_labels.detach().cpu().numpy().flatten()

                n_correct += np.sum(pred_flat == labels_flat)
                n_all += len(labels_flat)
                train_scores = n_correct / n_all

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                #pbar.set_postfix({'loss': total_loss/(sidx+1), 'accu': train_scores})
        
        torch.save(mlp.state_dict(), './cls/mlp_{}.pkl'.format(args.first_seed))
        
        mlp.eval()
        with torch.no_grad():
            if cuda:
                predict = mlp(embedding[test_mask].cuda())
            else:
                predict = mlp(embedding[test_mask])

            #predict = predict.cpu().detach().numpy()
            #predict = np.argmax(predict, axis=1).flatten()
            #scores = f1_score(pred_flat, labels[test_mask].flatten(), average='macro')
            #scores = calculate_scores(test_label.flatten(), predict, mode=args.task)
            acc = accuracy(predict, labels[test_mask])
            print("MLP training, acc: {}".format(acc))
            row_0['mlp'].append(acc)

        # svm linear
        lsvm = SVC(random_state=args.first_seed, kernel='linear')
        lsvm.fit(embedding[train_mask], labels[train_mask])
        predict = lsvm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Linear SVM training, acc: {}".format(acc))
        with open('./cls/lsvm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(lsvm, f)
        row_0['lsvm'].append(acc)
        
        # svm rbf
        r_svm = SVC(random_state=args.first_seed, kernel='rbf')
        r_svm.fit(embedding[train_mask], labels[train_mask])
        predict = r_svm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("RBF SVM training, acc: {}".format(acc))
        with open('./cls/rbf_svm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(r_svm, f)
        row_0['rsvm'].append(acc)
                
        # decision tree
        dt = DecisionTreeClassifier(random_state=args.first_seed, criterion='entropy')
        dt.fit(embedding[train_mask], labels[train_mask])
        predict = dt.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Decision tree training, acc: {}".format(acc))
        with open('./cls/decision_tree_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(dt, f)
        row_0['dt'].append(acc)

                
        # random forest
        #dt = RandomForestClassifier(n_estimators = 100, random_state=args.first_seed, criterion='entropy')
        rf = RandomForestClassifier(random_state=args.first_seed, criterion='entropy')
        rf.fit(embedding[train_mask], labels[train_mask])
        predict = rf.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Random forest training, acc: {}".format(acc))
        with open('./cls/random_forest_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(rf, f)
        row_0['rf'].append(acc)

        # lgbm
        clf_lgbm = LGBMClassifier(random_state=args.first_seed, n_jobs=16)
        clf_lgbm.fit(embedding[train_mask], labels[train_mask])
        clf_lgbm.fit(
            embedding[train_mask],
            labels[train_mask],
            eval_set=[(embedding[test_mask], labels[test_mask])],
            early_stopping_rounds=100,
            verbose=100,
        )
        predict = clf_lgbm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Light GBM training, acc: {}".format(acc))
        with open('./cls/lbgm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(clf_lgbm, f)
        row_0['lgbm'].append(acc)

        
        '''
        evaluate on fix cls
        '''

        random.seed(args.second_seed)
        np.random.seed(args.second_seed)
        torch.manual_seed(args.second_seed)

        # Create graph
        g = data.graph
        # add self loop
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        n_edges = g.number_of_edges()
        # create model

        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model_fix = GAT(g,
                    args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
        #print(model_fix)
        if args.early_stop:
            stopper = EarlyStopping(patience=100)
        if cuda:
            model_fix = model_fix.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # load last layer
        model_fix.gat_layers[-1] = model.gat_layers[-1]
        for param in model_fix.gat_layers[-1].parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            model_fix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            model_fix.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits, embedding = model_fix(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[train_mask], labels[train_mask])

            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate(model_fix, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model_fix):   
                        break

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                format(epoch, np.mean(dur), loss.item(), train_acc,
                        val_acc, n_edges / np.mean(dur) / 1000))

        print()
        #if args.early_stop:
        #    model.load_state_dict(torch.load('es_checkpoint.pt'))
        #acc = evaluate(model, features, labels, test_mask)
        #print("Test Accuracy {:.4f}".format(acc))

        # save model
        torch.save(model_fix.state_dict(), './model/fix_model_seed{}.pkl'.format(args.second_seed))

        # Train downstream cls
        print("\nSecond model - Fix, seed: ", args.second_seed)

        # Get embedding
        with torch.no_grad():
            logits, embedding = model_fix(features)

            # GAT
            logits = model.gat_layers[-1](model.g, embedding).mean(1)
            acc = accuracy(logits[test_mask], labels[test_mask])
            print("GAT cls Test Accuracy {:.4f}".format(acc))
            fix['gat'].append(acc)

        # linear
        perceptron.eval()
        with torch.no_grad():
            if cuda:
                predict = perceptron(embedding[test_mask].cuda())
            else:
                predict = perceptron(embedding[test_mask])

            #predict = predict.cpu().detach().numpy()
            #predict = np.argmax(predict, axis=1).flatten()
            #scores = f1_score(pred_flat, labels[test_mask].flatten(), average='macro')
            #scores = calculate_scores(test_label.flatten(), predict, mode=args.task)
            acc = accuracy(predict, labels[test_mask])
            print("Perceptron evaluation, acc: {}".format(acc))
            fix['perceptron'].append(acc)

        # MLP
        mlp.eval()
        with torch.no_grad():
            if cuda:
                predict = mlp(embedding[test_mask].cuda())
            else:
                predict = mlp(embedding[test_mask])

            #predict = predict.cpu().detach().numpy()
            #predict = np.argmax(predict, axis=1).flatten()
            #scores = f1_score(pred_flat, labels[test_mask].flatten(), average='macro')
            #scores = calculate_scores(test_label.flatten(), predict, mode=args.task)
            acc = accuracy(predict, labels[test_mask])
            print("MLP evaluation, acc: {}".format(acc))
            fix['mlp'].append(acc)

        # svm linear
        predict = lsvm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Linear SVM evaluation, acc: {}".format(acc))
        with open('./cls/lsvm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(lsvm, f)
        fix['lsvm'].append(acc)
        
        # svm rbf
        predict = r_svm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("RBF SVM evaluation, acc: {}".format(acc))
        with open('./cls/rbf_svm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(r_svm, f)
        fix['rsvm'].append(acc)
                
        # decision tree
        predict = dt.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Decision tree evaluation, acc: {}".format(acc))
        with open('./cls/decision_tree_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(dt, f)
        fix['dt'].append(acc)
                
        # random forest
        predict = rf.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Random forest evaluation, acc: {}".format(acc))
        with open('./cls/random_forest_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(rf, f)
        fix['rf'].append(acc)

        # lgbm
        predict = clf_lgbm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Light GBM evaluation, acc: {}".format(acc))
        with open('./cls/lbgm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(clf_lgbm, f)
        fix['lgbm'].append(acc)

        
        '''
        evaluate w/o fix cls
        '''

        random.seed(args.second_seed)
        np.random.seed(args.second_seed)
        torch.manual_seed(args.second_seed)

        # Create graph
        g = data.graph
        # add self loop
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        n_edges = g.number_of_edges()
        # create model

        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model_nofix = GAT(g,
                    args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
        #print(model_fix)
        if args.early_stop:
            stopper = EarlyStopping(patience=100)
        if cuda:
            model_nofix.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # no fix

        optimizer = torch.optim.Adam(
            model_nofix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            model_nofix.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits, embedding = model_nofix(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[train_mask], labels[train_mask])

            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate(model_fix, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model_fix):   
                        break

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                format(epoch, np.mean(dur), loss.item(), train_acc,
                        val_acc, n_edges / np.mean(dur) / 1000))

        print()
        #if args.early_stop:
        #    model.load_state_dict(torch.load('es_checkpoint.pt'))
        #acc = evaluate(model, features, labels, test_mask)
        #print("Test Accuracy {:.4f}".format(acc))

        # save model
        torch.save(model_fix.state_dict(), './model/nofix_model_seed{}.pkl'.format(args.second_seed))

        # Train downstream cls
        print("\nSecond model - w/o Fix, seed: ", args.second_seed)

        # Get embedding
        with torch.no_grad():
            logits, embedding = model_nofix(features)

            # GAT
            logits = model.gat_layers[-1](model.g, embedding).mean(1)
            acc = accuracy(logits[test_mask], labels[test_mask])
            print("Cls Test Accuracy {:.4f}".format(acc))
            no_fix['gat'].append(acc)

        # linear
        perceptron.eval()
        with torch.no_grad():
            if cuda:
                predict = perceptron(embedding[test_mask].cuda())
            else:
                predict = perceptron(embedding[test_mask])

            #predict = predict.cpu().detach().numpy()
            #predict = np.argmax(predict, axis=1).flatten()
            #scores = f1_score(pred_flat, labels[test_mask].flatten(), average='macro')
            #scores = calculate_scores(test_label.flatten(), predict, mode=args.task)
            acc = accuracy(predict, labels[test_mask])
            print("Perceptron evaluation, acc: {}".format(acc))
            no_fix['perceptron'].append(acc)

        # MLP
        mlp.eval()
        with torch.no_grad():
            if cuda:
                predict = mlp(embedding[test_mask].cuda())
            else:
                predict = mlp(embedding[test_mask])

            #predict = predict.cpu().detach().numpy()
            #predict = np.argmax(predict, axis=1).flatten()
            #scores = f1_score(pred_flat, labels[test_mask].flatten(), average='macro')
            #scores = calculate_scores(test_label.flatten(), predict, mode=args.task)
            acc = accuracy(predict, labels[test_mask])
            print("MLP evaluation, acc: {}".format(acc))
            no_fix['mlp'].append(acc)

        # svm linear
        predict = lsvm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Linear SVM evaluation, acc: {}".format(acc))
        with open('./cls/lsvm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(lsvm, f)
        no_fix['lsvm'].append(acc)
        
        # svm rbf
        predict = r_svm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("RBF SVM evaluation, acc: {}".format(acc))
        with open('./cls/rbf_svm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(r_svm, f)
        no_fix['rsvm'].append(acc)
                
        # decision tree
        predict = dt.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Decision tree evaluation, acc: {}".format(acc))
        with open('./cls/decision_tree_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(dt, f)
        no_fix['dt'].append(acc)
                
        # random forest
        predict = rf.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Random forest evaluation, acc: {}".format(acc))
        with open('./cls/random_forest_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(rf, f)
        no_fix['rf'].append(acc)

        # lgbm
        predict = clf_lgbm.predict(embedding[test_mask])
        predict = torch.Tensor(predict)
        acc = accuracy(predict, labels[test_mask])
        print("Light GBM evaluation, acc: {}".format(acc))
        with open('./cls/lbgm_{}.pickle'.format(args.first_seed), 'wb') as f:
            pickle.dump(clf_lgbm, f)
        no_fix['lgbm'].append(acc)


    # show result
    logging.info("\n\n -- END -- ")
    logging.info("For GAT - Cora node cls: #Sampling: {}".format(args.num_sampling))

    gat_m, gat_h = mean_confidence_interval(row_0['gat'])
    perceptron_m, perceptron_h = mean_confidence_interval(row_0['perceptron'])
    mlp_m, mlp_h = mean_confidence_interval(row_0['mlp'])
    lsvm_m, lsvm_h = mean_confidence_interval(row_0['lsvm'])
    svm_m, svm_h = mean_confidence_interval(row_0['rsvm'])
    dt_m, dt_h = mean_confidence_interval(row_0['dt'])
    rf_m, rf_h = mean_confidence_interval(row_0['rf'])
    lgbm_m, lgbm_h = mean_confidence_interval(row_0['lgbm'])

    print(f"Row 0: GAT {gat_m*100:.5f} +- {gat_h*100:.5f}," 
        f"perceptron {perceptron_m*100:.5f} +- {perceptron_h*100:.5f}, "
        f"mlp {mlp_m*100:.5f} +- {mlp_h*100:.5f}, "
        f"lsvm {lsvm_m*100:.5f} +- {lsvm_h*100:.5f}, "
        f"rbf_svm {svm_m*100:.5f} +- {svm_h*100:.5f}, "
        f"dt {dt_m*100:.5f} +- {dt_h*100:.5f}, "
        f"rf {rf_m*100:.5f} +- {rf_h*100:.5f}, "
        f"lgbm {lgbm_m*100:.5f} +- {lgbm_h*100:.5f}")

    logging.info(f"Row 0: GAT {gat_m*100:.5f} +- {gat_h*100:.5f}," 
                f"perceptron {perceptron_m*100:.5f} +- {perceptron_h*100:.5f}, "
                f"mlp {mlp_m*100:.5f} +- {mlp_h*100:.5f}, "
                f"lsvm {lsvm_m*100:.5f} +- {lsvm_h*100:.5f}, "
                f"rbf_svm {svm_m*100:.5f} +- {svm_h*100:.5f}, "
                f"dt {dt_m*100:.5f} +- {dt_h*100:.5f}, "
                f"rf {rf_m*100:.5f} +- {rf_h*100:.5f}, "
                f"lgbm {lgbm_m*100:.5f} +- {lgbm_h*100:.5f}")
    
    gat_m, gat_h = mean_confidence_interval(fix['gat'])
    perceptron_m, perceptron_h = mean_confidence_interval(fix['perceptron'])
    mlp_m, mlp_h = mean_confidence_interval(fix['mlp'])
    lsvm_m, lsvm_h = mean_confidence_interval(fix['lsvm'])
    svm_m, svm_h = mean_confidence_interval(fix['rsvm'])
    dt_m, dt_h = mean_confidence_interval(fix['dt'])
    rf_m, rf_h = mean_confidence_interval(fix['rf'])
    lgbm_m, lgbm_h = mean_confidence_interval(fix['lgbm'])

    print(f"Fix: GAT {gat_m*100:.5f} +- {gat_h*100:.5f}," 
        f"perceptron {perceptron_m*100:.5f} +- {perceptron_h*100:.5f}, "
        f"mlp {mlp_m*100:.5f} +- {mlp_h*100:.5f}, "
        f"lsvm {lsvm_m*100:.5f} +- {lsvm_h*100:.5f}, "
        f"rbf_svm {svm_m*100:.5f} +- {svm_h*100:.5f}, "
        f"dt {dt_m*100:.5f} +- {dt_h*100:.5f}, "
        f"rf {rf_m*100:.5f} +- {rf_h*100:.5f}, "
        f"lgbm {lgbm_m*100:.5f} +- {lgbm_h*100:.5f}")
    
    logging.info(f"Fix: GAT {gat_m*100:.5f} +- {gat_h*100:.5f}," 
                f"perceptron {perceptron_m*100:.5f} +- {perceptron_h*100:.5f}, "
                f"mlp {mlp_m*100:.5f} +- {mlp_h*100:.5f}, "
                f"lsvm {lsvm_m*100:.5f} +- {lsvm_h*100:.5f}, "
                f"rbf_svm {svm_m*100:.5f} +- {svm_h*100:.5f}, "
                f"dt {dt_m*100:.5f} +- {dt_h*100:.5f}, "
                f"rf {rf_m*100:.5f} +- {rf_h*100:.5f}, "
                f"lgbm {lgbm_m*100:.5f} +- {lgbm_h*100:.5f}")

    gat_m, gat_h = mean_confidence_interval(no_fix['gat'])
    perceptron_m, perceptron_h = mean_confidence_interval(no_fix['perceptron'])
    mlp_m, mlp_h = mean_confidence_interval(no_fix['mlp'])
    lsvm_m, lsvm_h = mean_confidence_interval(no_fix['lsvm'])
    svm_m, svm_h = mean_confidence_interval(no_fix['rsvm'])
    dt_m, dt_h = mean_confidence_interval(no_fix['dt'])
    rf_m, rf_h = mean_confidence_interval(no_fix['rf'])
    lgbm_m, lgbm_h = mean_confidence_interval(no_fix['lgbm'])

    print(f"W/O fix cls: GAT {gat_m*100:.5f} +- {gat_h*100:.5f}," 
        f"perceptron {perceptron_m*100:.5f} +- {perceptron_h*100:.5f}, "
        f"mlp {mlp_m*100:.5f} +- {mlp_h*100:.5f}, "
        f"lsvm {lsvm_m*100:.5f} +- {lsvm_h*100:.5f}, "
        f"rbf_svm {svm_m*100:.5f} +- {svm_h*100:.5f}, "
        f"dt {dt_m*100:.5f} +- {dt_h*100:.5f}, "
        f"rf {rf_m*100:.5f} +- {rf_h*100:.5f}, "
        f"lgbm {lgbm_m*100:.5f} +- {lgbm_h*100:.5f}")

    logging.info(f"W/O fix cls: GAT {gat_m*100:.5f} +- {gat_h*100:.5f}," 
                f"perceptron {perceptron_m*100:.5f} +- {perceptron_h*100:.5f}, "
                f"mlp {mlp_m*100:.5f} +- {mlp_h*100:.5f}, "
                f"lsvm {lsvm_m*100:.5f} +- {lsvm_h*100:.5f}, "
                f"rbf_svm {svm_m*100:.5f} +- {svm_h*100:.5f}, "
                f"dt {dt_m*100:.5f} +- {dt_h*100:.5f}, "
                f"rf {rf_m*100:.5f} +- {rf_h*100:.5f}, "
                f"lgbm {lgbm_m*100:.5f} +- {lgbm_h*100:.5f}")


    # save
    cls_name = ['row0', 'fix', 'no_fix']
    for i, s in enumerate([row_0, fix, no_fix]):
        with open('./results/GAT_{}_results({}_times).json'.format(cls_name[i], args.num_sampling), 'w') as f:
            json.dump(s, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")

    parser.add_argument('--train_cls', action="store_true", default=False,
                        help="use pre-trained embedding to train cls")
    parser.add_argument('--fix_cls', action="store_true", default=False,
                        help="fix cls to test embedding")                        
    parser.add_argument('--load_row0_cls', action="store_true", default=False,
                        help="load row 0's cls weight to train another set of embedding")       
    parser.add_argument('--seed', type=int, default=666,
                        help="set seed")
    parser.add_argument('--first_seed', type=int, default=1,
                        help="set first seed")
    parser.add_argument('--second_seed', type=int, default=1,
                        help="set first seed")
    parser.add_argument('--save_model', action="store_true", default=False,
                        help="save model")  

    parser.add_argument('--num_sampling', type=int, default=15,
                        help="Number of run")
    parser.add_argument('--num_mlp_loop', type=int, default=100,
                        help="Number of MLP/Perceptron run")        
    parser.add_argument('--logfile', default='log.log',
                        help='Logging file')   
    parser.add_argument('--check_model', action="store_true", default=False,
                        help="save model")                                                   
    

    args = parser.parse_args()
    print(args)

    main(args)
