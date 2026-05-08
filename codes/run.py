#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel
from metrics import classification_metrics

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from dataloader import read_labeled_triple

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    parser.add_argument('--model_type', type=str, default='entity_relation_embedding')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('--gamma_uni', default=1.0, type=float)
    parser.add_argument('--gamma_neg', default=1.0, type=float)
    parser.add_argument('--epsilon', default=1e-8, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    # Fallbacks support older checkpoints that may not store these fields.
    args.double_entity_embedding = argparse_dict.get(
        'double_entity_embedding',
        args.model in ['RotatE', 'ComplEx', 'DirectAU_RotatE']
    )
    args.double_relation_embedding = argparse_dict.get(
        'double_relation_embedding',
        args.model == 'ComplEx'
    )
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.model_type = argparse_dict.get('model_type', args.model_type)
    args.gamma_uni = argparse_dict.get('gamma_uni', args.gamma_uni)
    args.gamma_neg = argparse_dict.get('gamma_neg', args.gamma_neg)
    args.epsilon = argparse_dict.get('epsilon', args.epsilon)


def get_dataset_name(args):
    return os.path.basename(os.path.normpath(args.data_path))


def resolve_labeled_data_path(args):
    dataset_name = get_dataset_name(args)
    if dataset_name.endswith('_w_label') or dataset_name.endswith('_w_labels'):
        if os.path.exists(args.data_path):
            return args.data_path

    dataset_root = os.path.dirname(os.path.normpath(args.data_path))
    candidates = [
        os.path.join(dataset_root, dataset_name + '_w_labels'),
        os.path.join(dataset_root, dataset_name + '_w_label'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def format_link_prediction_metrics(metrics):
    formatted = {}
    if 'MR' in metrics:
        formatted['MR'] = metrics['MR']
    if 'MRR' in metrics:
        formatted['MRR'] = metrics['MRR']
    if 'HITS@1' in metrics:
        formatted['Hit@1'] = metrics['HITS@1']
    if 'HITS@3' in metrics:
        formatted['Hit@3'] = metrics['HITS@3']
    if 'HITS@10' in metrics:
        formatted['Hit@10'] = metrics['HITS@10']
    if len(formatted) == 0:
        return metrics
    return formatted


def evaluate_triple_classification(model, labeled_triples, args):
    if labeled_triples is None or len(labeled_triples) == 0:
        return None

    model.eval()
    scores = []
    predictions = []
    labels = []

    batch_size = max(1, args.test_batch_size)

    with torch.no_grad():
        for start in range(0, len(labeled_triples), batch_size):
            chunk = labeled_triples[start:start + batch_size]
            sample = torch.LongTensor([(h, r, t) for h, r, t, _ in chunk])
            if args.cuda:
                sample = sample.cuda()

            chunk_scores = model(sample).view(-1).cpu().numpy().tolist()
            scores.extend(chunk_scores)

            for (_, _, _, label), score in zip(chunk, chunk_scores):
                labels.append(label)
                predictions.append(1 if score > 0 else 0)

    metrics = classification_metrics(predictions, labels, scores)
    return {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'PR-AUC': metrics['pr_auc'],
        'ROC-AUC': metrics['roc_auc'],
    }
    
def save_model(model, optimizer, save_variable_list, args, checkpoint_dir=None, model_filename=None):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    checkpoint_dir = checkpoint_dir or args.save_path
    model_filename = model_filename or '%s_%s.mdl' % (args.run_timestamp, args.model)

    argparse_dict = vars(args)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(checkpoint_dir, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(checkpoint_dir, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(checkpoint_dir, 'relation_embedding'), 
        relation_embedding
    )

    if hasattr(model, 'relation_mask_embedding') and model.relation_mask_embedding is not None:
        relation_mask_embedding = model.relation_mask_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(checkpoint_dir, 'relation_mask_embedding'),
            relation_mask_embedding
        )

    model_file = os.path.join(
        args.model_save_path,
        model_filename
    )
    torch.save(
        {
            **save_variable_list,
            'model_state_dict': model.state_dict(),
        },
        model_file
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_triple_with_label(file_path, entity2id, relation2id):
    '''
    Read triples and label, map them into ids and int label.
    Return: list of (h, r, t, label)
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                h, r, t, label = parts
                triples.append((entity2id[h], relation2id[r], entity2id[t], int(label)))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    os.makedirs(args.log_save_path, exist_ok=True)
    log_file = os.path.join(
        args.log_save_path,
        '%s_%s.log' % (args.run_timestamp, args.model)
    )

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def log_final_summary(args, link_metrics=None, cls_metrics=None, training_time_sec=0.0,
                      total_runtime_sec=0.0, best_valid_epoch=None, best_valid_mrr=None):
    logging.info('================ Final Result ================')
    logging.info('Model name: %s' % args.model)

    if link_metrics is not None:
        logging.info('MR: %s' % link_metrics.get('MR', 'N/A'))
        logging.info('MRR: %s' % link_metrics.get('MRR', 'N/A'))
        logging.info('Hit@1: %s' % link_metrics.get('Hit@1', 'N/A'))
        logging.info('Hit@3: %s' % link_metrics.get('Hit@3', 'N/A'))
        logging.info('Hit@10: %s' % link_metrics.get('Hit@10', 'N/A'))
    else:
        logging.info('MR: N/A')
        logging.info('MRR: N/A')
        logging.info('Hit@1: N/A')
        logging.info('Hit@3: N/A')
        logging.info('Hit@10: N/A')

    if cls_metrics is not None:
        logging.info('Acc: %s' % cls_metrics.get('Accuracy', 'N/A'))
        logging.info('Prec: %s' % cls_metrics.get('Precision', 'N/A'))
        logging.info('Rec: %s' % cls_metrics.get('Recall', 'N/A'))
        logging.info('F1: %s' % cls_metrics.get('F1 Score', 'N/A'))
        logging.info('PR-AUC: %s' % cls_metrics.get('PR-AUC', 'N/A'))
        logging.info('ROC-AUC: %s' % cls_metrics.get('ROC-AUC', 'N/A'))
    else:
        logging.info('Acc: N/A')
        logging.info('Prec: N/A')
        logging.info('Rec: N/A')
        logging.info('F1: N/A')
        logging.info('PR-AUC: N/A')
        logging.info('ROC-AUC: N/A')

    logging.info('Training time: %.3f sec' % training_time_sec)
    logging.info('Total time: %.3f sec' % total_runtime_sec)

    if best_valid_epoch is None or best_valid_epoch < 0:
        logging.info('Best Valid Epoch: N/A')
        logging.info('Best Valid MRR: N/A')
    else:
        logging.info('Best Valid Epoch: %d' % best_valid_epoch)
        logging.info('Best Valid MRR: %.6f' % best_valid_mrr)

    logging.info('=============================================')
        
        
def main(args):
    total_start_time = time.time()
    train_start_time = None

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    args.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = get_dataset_name(args)
    args.log_save_path = os.path.join('logs', dataset_name)
    args.model_save_path = os.path.join('models', dataset_name)
    args.best_save_path = os.path.join(args.save_path, 'best') if args.save_path else None
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.best_save_path:
        os.makedirs(args.best_save_path, exist_ok=True)

    os.makedirs(args.log_save_path, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))

    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    labeled_data_path = resolve_labeled_data_path(args)
    valid_labeled_triples = None
    test_labeled_triples = None
    if labeled_data_path:
        valid_labeled_path = os.path.join(labeled_data_path, 'valid.txt')
        test_labeled_path = os.path.join(labeled_data_path, 'test.txt')

        if os.path.exists(valid_labeled_path):
            valid_labeled_triples = read_labeled_triple(valid_labeled_path, entity2id, relation2id)
            logging.info('#valid_labeled: %d' % len(valid_labeled_triples))

        if os.path.exists(test_labeled_path):
            test_labeled_triples = read_labeled_triple(test_labeled_path, entity2id, relation2id)
            logging.info('#test_labeled: %d' % len(test_labeled_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        train_start_time = time.time()
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=min(4, max(1, args.cpu_num//2)),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=min(4, max(1, args.cpu_num//2)),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint_path = os.path.join(args.init_checkpoint, 'checkpoint')
        best_checkpoint_path = os.path.join(args.init_checkpoint, 'best', 'checkpoint')
        if (not args.do_train) and os.path.exists(best_checkpoint_path):
            checkpoint_path = best_checkpoint_path
            logging.info('Using best checkpoint %s...' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    if args.model == 'DirectAU_RotatE':
        logging.info('gamma_uni = %f' % args.gamma_uni)
        logging.info('gamma_neg = %f' % args.gamma_neg)
        logging.info('epsilon = %e' % args.epsilon)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []
        training_logs_for_valid = []
        
        best_valid_step = -1
        best_valid_epoch = -1
        best_valid_mrr = -1.0
        validation_round = 0

        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            training_logs_for_valid.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                validation_round += 1
                # Log aggregated component losses for this training epoch (since last validation)
                if len(training_logs_for_valid) > 0:
                    epoch_metrics = {}
                    for metric in training_logs_for_valid[0].keys():
                        epoch_metrics[metric] = sum([l[metric] for l in training_logs_for_valid]) / len(training_logs_for_valid)
                    log_metrics('Training epoch average', step, epoch_metrics)
                    training_logs_for_valid = []

                logging.info('Evaluating on Valid Dataset...')
                link_metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid Link Prediction', step, format_link_prediction_metrics(link_metrics))

                current_valid_mrr = link_metrics.get('MRR')
                if current_valid_mrr is not None and current_valid_mrr > best_valid_mrr:
                    best_valid_mrr = current_valid_mrr
                    best_valid_step = step
                    best_valid_epoch = validation_round
                    if args.best_save_path:
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'best_valid_step': best_valid_step,
                            'best_valid_epoch': best_valid_epoch,
                            'best_valid_mrr': best_valid_mrr,
                        }
                        save_model(
                            kge_model,
                            optimizer,
                            save_variable_list,
                            args,
                            checkpoint_dir=args.best_save_path,
                            model_filename='%s_%s_best.mdl' % (args.run_timestamp, args.model)
                        )

                cls_metrics = evaluate_triple_classification(kge_model, valid_labeled_triples, args)
                if cls_metrics is not None:
                    log_metrics('Valid Triple Classification', step, cls_metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

        training_time_sec = time.time() - train_start_time
        logging.info('Training time (sec): %.3f' % training_time_sec)
        if best_valid_step >= 0:
            logging.info('Best valid epoch: %d' % best_valid_epoch)
            logging.info('Best valid step: %d' % best_valid_step)
            logging.info('Best valid MRR: %.6f' % best_valid_mrr)
        else:
            logging.info('Best valid epoch: N/A')
            logging.info('Best valid step: N/A')
            logging.info('Best valid MRR: N/A')
    else:
        training_time_sec = 0.0
        best_valid_epoch = -1
        best_valid_mrr = None

    final_link_metrics = None
    final_cls_metrics = None

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        link_metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid Link Prediction', step, format_link_prediction_metrics(link_metrics))
        final_link_metrics = format_link_prediction_metrics(link_metrics)

        cls_metrics = evaluate_triple_classification(kge_model, valid_labeled_triples, args)
        if cls_metrics is not None:
            log_metrics('Valid Triple Classification', step, cls_metrics)
            final_cls_metrics = cls_metrics
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        link_metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test Link Prediction', step, format_link_prediction_metrics(link_metrics))
        final_link_metrics = format_link_prediction_metrics(link_metrics)

        cls_metrics = evaluate_triple_classification(kge_model, test_labeled_triples, args)
        if cls_metrics is not None:
            log_metrics('Test Triple Classification', step, cls_metrics)
            final_cls_metrics = cls_metrics
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    total_runtime_sec = time.time() - total_start_time
    logging.info('Total running time (sec): %.3f' % total_runtime_sec)

    log_final_summary(
        args,
        link_metrics=final_link_metrics,
        cls_metrics=final_cls_metrics,
        training_time_sec=training_time_sec if args.do_train else 0.0,
        total_runtime_sec=total_runtime_sec,
        best_valid_epoch=best_valid_epoch,
        best_valid_mrr=best_valid_mrr,
    )
        
if __name__ == '__main__':
    main(parse_args())
