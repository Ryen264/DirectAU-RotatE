#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.relation_mask_embedding = None
        if model_name == 'DirectAU_RotatE':
            self.relation_mask_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
            nn.init.uniform_(
                tensor=self.relation_mask_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'DirectAU_RotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'DirectAU_RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('DirectAU_RotatE should use --double_entity_embedding and must not use --double_relation_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def _normalize_complex_embedding(self, embedding):
        re_part, im_part = torch.chunk(embedding, 2, dim=-1)
        norm = torch.sqrt(re_part.pow(2) + im_part.pow(2)).clamp_min(1e-12)
        return re_part / norm, im_part / norm

    def _normalize_complex_pair(self, re_part, im_part):
        norm = torch.sqrt(re_part.pow(2) + im_part.pow(2)).clamp_min(1e-12)
        return re_part / norm, im_part / norm

    def _complex_to_real(self, re_part, im_part):
        return torch.cat([re_part, im_part], dim=-1)

    def _relation_phase(self, relation_embedding):
        return torch.sigmoid(relation_embedding) * (2.0 * np.pi)

    def _relation_unit(self, relation_embedding):
        phase = self._relation_phase(relation_embedding)
        return torch.cos(phase), torch.sin(phase)

    def _relation_mask(self, relation_ids):
        relation_mask = torch.index_select(
            self.relation_mask_embedding,
            dim=0,
            index=relation_ids
        )
        return torch.sigmoid(relation_mask).unsqueeze(1)

    def _compose_query_from_head(self, head, relation_embedding, relation_ids):
        head_re, head_im = self._normalize_complex_embedding(head)
        relation_mask = self._relation_mask(relation_ids)
        head_re = head_re * relation_mask
        head_im = head_im * relation_mask

        relation_re, relation_im = self._relation_unit(relation_embedding)
        query_re = head_re * relation_re - head_im * relation_im
        query_im = head_re * relation_im + head_im * relation_re
        return self._normalize_complex_pair(query_re, query_im)

    def _compose_query_from_tail(self, tail, relation_embedding, relation_ids):
        tail_re, tail_im = self._normalize_complex_embedding(tail)
        relation_mask = self._relation_mask(relation_ids)
        tail_re = tail_re * relation_mask
        tail_im = tail_im * relation_mask

        relation_re, relation_im = self._relation_unit(relation_embedding)
        query_re = tail_re * relation_re + tail_im * relation_im
        query_im = -tail_re * relation_im + tail_im * relation_re
        return self._normalize_complex_pair(query_re, query_im)

    def _score_complex(self, lhs_re, lhs_im, rhs_re, rhs_im):
        return (lhs_re * rhs_re + lhs_im * rhs_im).sum(dim=-1)

    def DirectAU_RotatE(self, head, relation, tail, relation_ids, mode):
        if mode in ['single', 'tail-batch']:
            query_re, query_im = self._compose_query_from_head(head, relation, relation_ids)
            candidate_re, candidate_im = self._normalize_complex_embedding(tail)
        elif mode == 'head-batch':
            query_re, query_im = self._compose_query_from_tail(tail, relation, relation_ids)
            candidate_re, candidate_im = self._normalize_complex_embedding(head)
        else:
            raise ValueError('mode %s not supported' % mode)

        return self._score_complex(query_re, query_im, candidate_re, candidate_im)
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            relation_ids = sample[:, 1]
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            relation_ids = tail_part[:, 1]
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            relation_ids = head_part[:, 1]
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name == 'DirectAU_RotatE':
            score = self.DirectAU_RotatE(head, relation, tail, relation_ids, mode)
        elif self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def _directau_training_step(self, optimizer, train_iterator, args):
        self.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=positive_sample[:, 0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=positive_sample[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=positive_sample[:, 2]
        ).unsqueeze(1)

        relation_ids = positive_sample[:, 1]

        align_query_re, align_query_im = self._compose_query_from_head(head, relation, relation_ids)
        align_target_re, align_target_im = self._normalize_complex_embedding(tail)
        align_query = self._complex_to_real(align_query_re, align_query_im)
        align_target = self._complex_to_real(align_target_re, align_target_im)
        align_sample_loss = (align_query - align_target).pow(2).mean(dim=-1).squeeze(-1)

        if mode == 'head-batch':
            negative_query_re, negative_query_im = self._compose_query_from_tail(tail, relation, relation_ids)
        elif mode == 'tail-batch':
            negative_query_re, negative_query_im = self._compose_query_from_head(head, relation, relation_ids)
        else:
            raise ValueError('mode %s not supported' % mode)

        all_entity_ids = torch.unique(torch.cat([positive_sample[:, 0], positive_sample[:, 2]], dim=0))
        if all_entity_ids.numel() < 2:
            uniformity_loss = torch.zeros(1, device=positive_sample.device).squeeze(0)
        else:
            uniform_entities = torch.index_select(self.entity_embedding, dim=0, index=all_entity_ids)
            uniform_entities = self._complex_to_real(*self._normalize_complex_embedding(uniform_entities))
            pairwise_distance = torch.pdist(uniform_entities, p=2)
            if pairwise_distance.numel() == 0:
                uniformity_loss = torch.zeros(1, device=positive_sample.device).squeeze(0)
            else:
                uniformity_loss = torch.log(
                    torch.mean(torch.exp(-2.0 * pairwise_distance.pow(2))) + args.epsilon
                )

        if args.uni_weight:
            align_loss = align_sample_loss.mean()
            per_sample_weight = torch.full_like(subsampling_weight, 1.0 / subsampling_weight.numel())
        else:
            subsampling_weight = subsampling_weight.view(-1)
            align_loss = (subsampling_weight * align_sample_loss).sum() / subsampling_weight.sum()
            per_sample_weight = subsampling_weight / subsampling_weight.sum()

        base_loss = align_loss + args.gamma_uni * uniformity_loss
        base_loss.backward(retain_graph=args.gamma_neg > 0.0)

        negative_sample_loss = torch.zeros(
            negative_sample.size(0),
            device=positive_sample.device,
            dtype=self.entity_embedding.dtype
        )

        if args.gamma_neg > 0.0:
            negative_chunk_size = max(1, int(getattr(args, 'directau_negative_chunk_size', 32)))
            total_negative_count = max(negative_sample.size(1), 1)
            negative_scale = args.gamma_neg / total_negative_count

            for start in range(0, negative_sample.size(1), negative_chunk_size):
                end = min(start + negative_chunk_size, negative_sample.size(1))
                chunk_negative = negative_sample[:, start:end]
                chunk_candidate = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=chunk_negative.reshape(-1)
                ).view(negative_sample.size(0), end - start, -1)

                chunk_candidate_re, chunk_candidate_im = self._normalize_complex_embedding(chunk_candidate)
                chunk_score = self._score_complex(
                    negative_query_re,
                    negative_query_im,
                    chunk_candidate_re,
                    chunk_candidate_im
                )

                chunk_loss = F.softplus(chunk_score).sum(dim=1)
                negative_sample_loss += chunk_loss.detach() / total_negative_count

                chunk_loss_scalar = (per_sample_weight * chunk_loss).sum() * negative_scale
                chunk_loss_scalar.backward(retain_graph=(end < negative_sample.size(1)))

            if args.uni_weight:
                negative_sample_loss = negative_sample_loss.mean()
            else:
                negative_sample_loss = (subsampling_weight * negative_sample_loss).sum() / subsampling_weight.sum()
        else:
            negative_sample_loss = torch.zeros(1, device=positive_sample.device).squeeze(0)

        loss = align_loss + args.gamma_uni * uniformity_loss + args.gamma_neg * negative_sample_loss
        optimizer.step()

        return {
            'align_loss': align_loss.item(),
            'uniform_loss': uniformity_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        if model.model_name == 'DirectAU_RotatE':
            return model._directau_training_step(optimizer, train_iterator, args)

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
