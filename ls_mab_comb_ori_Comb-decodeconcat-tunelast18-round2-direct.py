#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    run.py
  Author:       locke
  Date created: 2020/3/25 下午6:58
"""
import time
import argparse 
import os
import pathlib
import gc
import random
import math
import numpy as np
import scipy.sparse as sp
import multiprocessing
from multiprocessing import Pool
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from models import *
from utils import *
import copy

# from torch.utils.tensorboard import SummaryWriter
# import logging
from sklearn.cluster import KMeans
import scipy


import sys, logging
import json, pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as ss
import faiss
from pathlib import Path
from random import shuffle
from sklearn import metrics
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import IterableDataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple


# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s %(levelname)s] - %(message)s'
# )
# logger = logging.getLogger(__name__)

import torch.nn.functional as F
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(device)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        pretrained_model,
        channels: int = 1,
    ):
        super(TransformerEncoder, self).__init__()
        self.transformers = torch.nn.ModuleList()
        for c in range(channels):
            self.transformers.append(AutoModel.from_pretrained(pretrained_model))

    def forward(
        self,
        tokens,
        channel,
    ):
        embeddings = self.transformers[channel](
            torch.squeeze(tokens['input_ids']).to(device),
            torch.squeeze(tokens['attention_mask']).to(device)
        )[0].to(device)
        return embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(device)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TwoTower(nn.Module):

    def __init__(
        self,
        pretrained_model,
    ):
        super(TwoTower, self).__init__()
        self.tower = TransformerEncoder(pretrained_model)
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, query_tokens, doc_tokens, labels):
        query_embeddings = self.tower(query_tokens,0)[:, 0].to(device)
        doc_embeddings = self.tower(doc_tokens,0)[:, 0].to(device)

        scores = torch.cosine_similarity(query_embeddings, doc_embeddings)
        # print("labels:",labels)
        # print("scores:",scores)
        # loss = self.loss_fn(scores, labels).sum()
        m = nn.Sigmoid()
        loss = nn.BCELoss()
        self.loss_fn = loss(m(scores), labels.to(device))
        loss = self.loss_fn.sum()
        
        return loss, scores


    def embeds(self, doc_tokens):
        # print("self.tower(doc_tokens,0)[:, 0]:", self.tower(doc_tokens,0)[:, 0])
        # print()
        return self.tower(doc_tokens,0).to(device) #[:, 0].to(device)

class FourTower(nn.Module):

    def __init__(
        self,
        pretrained_model,
        channels: int = 2,
    ):
        super(FourTower, self).__init__()
        self.towers = TransformerEncoder(pretrained_model, channels)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, query_tokens, q_meta_tokens, doc_tokens, d_meta_tokens, labels):
        query_embeddings = self.towers(query_tokens, 0)[:, 0]
        doc_embeddings = self.towers(doc_tokens, 0)[:, 0]
        q_meta_embeddings = self.towers(q_meta_tokens, 1)[:, 0]
        d_meta_embeddings = self.towers(d_meta_tokens, 1)[:, 0]

        scores = torch.cosine_similarity(
            (query_embeddings+q_meta_embeddings),
            (doc_embeddings+d_meta_embeddings)
        )
        loss = self.loss_fn(scores, labels).sum()
        return loss, scores


    def embeds(self, doc_tokens, meta_tokens):
        return self.towers(doc_tokens, 0)[:, 0]+self.towers(meta_tokens, 1)[:, 0]



def dump_code2tokens(
    kg_text_path: Path,
    # kg_path: Path,
    pretrained_model,
    pickle_path: Path,
    max_token_length = 64,
    device = 'cpu',
):
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = SentenceTransformer('bert-base-multilingual-cased', device=device)
    # model = SentenceTransformer('colorfulscoop/sbert-base-ja', device=device)
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

    logger.info('Load from')
    # logger.info(kg_path)
    e2m_list = defaultdict(list)
    # with kg_path.open('r') as f:
    #     for line in f:
    #         ent, meta, _ = line.rstrip('\n').split(' ')
    #         e2m_list[ent].append(meta)

    code2text = {}
    logger.info('Load from')
    logger.info(kg_text_path)
    with kg_text_path.open('r') as f:
        for line in tqdm(f):
            text, code = line.rstrip('\n').split('\t')
            code2text[str(code)] = text
    codes = [str(code) for code in list(code2text.keys())]# if code[0]=='e']

    code2tokens = {}
    # get embeddings
    logger.info('pre-compute embeddings')
    entity_texts = [code2text[code] for code in codes]
    # print("entity_texts:",len(entity_texts))
    # meta_texts = [
    #     ' '.join([code2text[m] for m in e2m_list[code]]) for code in codes]
    entity_embeds = np.array(model.encode(entity_texts), dtype='float32')
    # print("entity_embeds:",len(entity_embeds))
    # meta_embeds = np.array(model.encode(meta_texts), dtype='float32')
    # for code, e1, e2 in zip(codes, entity_embeds, meta_embeds):
    for code, e1 in zip(codes, entity_embeds):
        code2tokens[code] = {}
        code2tokens[code]['entity_embed'] = e1
        # code2tokens[code]['meta_embed'] = e2

    # get tokens
    logger.info('pre-compute tokenizations')
    for code in tqdm(codes, total=len(codes)):
        tokens = tokenizer(
            f"{code2text[code]}",
            truncation=True,
            max_length=max_token_length,
            padding='max_length',
            return_tensors="pt",
        )
        code2tokens[code]['entity_tokens'] = tokens

        # meta_text = ' '.join([code2text[m] for m in e2m_list[code]])
        # tokens = tokenizer(
        #     f"{meta_text}",
        #     truncation=True,
        #     max_length=max_token_length,
        #     padding='max_length',
        #     return_tensors="pt",
        # )
        # code2tokens[code]['meta_tokens'] = tokens

    logger.info('Save to')
    logger.info(pickle_path)
    pickle.dump(code2tokens, pickle_path.open('wb'))


class RawDataset(IterableDataset):

    def __init__(self):
        pass

    def __call__(
        self,
        code2tokens_path: Path,
    ):
        self.code2tokens = pickle.load(code2tokens_path.open('rb'))

        return self

    def __len__(self):
        return len(self.code2tokens)

    def __iter__(self):
        for code in self.code2tokens:
            yield \
                code, \
                self.code2tokens[code]['entity_tokens']

#####
class RawDataset_last(IterableDataset):

    def __init__(self):
        pass

    def __call__(
        self,
        code2tokens_path: Path,
    ):
        self.code2tokens = code2tokens_path

        return self

    def __len__(self):
        return len(self.code2tokens)

    def __iter__(self):
        for code in self.code2tokens:
            yield \
                code, \
                self.code2tokens[code]['entity_tokens']
#####

class PairsDataset(IterableDataset):

    def __init__(self):
        pass

    def __call__(
        self,
        train_pairs_path: Path,
        code2tokens_path: Path,
    ):
        logger.info('Load tokens/embeds')
        self.code2tokens = pickle.load(code2tokens_path.open('rb'))
        self.pairs = []
        with train_pairs_path.open('r') as f:
            for line in f:
                code1, code2 = line.rstrip('\n').split('\t')
                self.pairs.append((str(code1), str(code2)))

        logger.info('build ANN')
        self.codes = [code for code in self.code2tokens]
        self.index = {code:e for e, code in enumerate(self.codes)}
        self.entity_embeds = np.array(
            [self.code2tokens[code]['entity_embed'] for code in self.codes], dtype='float32')
        self.entity_nn = faiss.IndexFlatIP(len(self.entity_embeds[0]))
        self.entity_nn.add(self.entity_embeds)

        return self


    def __iter__(self):
        shuffle(self.pairs)
        rand_max = len(self.codes) - 1
        retrieval_max = 200
        for pair in self.pairs:
            # positive
            yield \
                self.code2tokens[pair[0]]['entity_tokens'], \
                self.code2tokens[pair[1]]['entity_tokens'], \
                1.0
            query_idx0 = self.index[pair[0]]
            query_idx1 = self.index[pair[1]]
            # entity-based negatives
            _, similars0 = self.entity_nn.search(
                self.entity_embeds[query_idx0:query_idx0+1], retrieval_max)
            _, similars1 = self.entity_nn.search(
                self.entity_embeds[query_idx1:query_idx1+1], retrieval_max)
            for _ in range(2): # hard negative
                rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                yield \
                    self.code2tokens[pair[0]]['entity_tokens'], \
                    self.code2tokens[rand_code]['entity_tokens'], \
                    0.0
            for _ in range(2): # easy negative
                rand_code = self.codes[random.randint(0, rand_max)]
                yield \
                    self.code2tokens[pair[0]]['entity_tokens'], \
                    self.code2tokens[rand_code]['entity_tokens'], \
                    0.0
            for _ in range(2): # hard negative
                rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                yield \
                    self.code2tokens[pair[1]]['entity_tokens'], \
                    self.code2tokens[rand_code]['entity_tokens'], \
                    0.0
            for _ in range(2): # easy negative
                rand_code = self.codes[random.randint(0, rand_max)]
                yield \
                    self.code2tokens[pair[1]]['entity_tokens'], \
                    self.code2tokens[rand_code]['entity_tokens'], \
                    0.0









pretrained_model ='bert-base-multilingual-cased' # 'sentence-transformers/all-mpnet-base-v2' #'bert-base-multilingual-cased' 'colorfulscoop/sbert-base-ja'
# pretrained_ckpt = str(0) #'' ##str(epoch-1) ##[For iter only]
# batch_size = 20
# model_path = Path('./sbert-fine-tune-model_status/'+dataset)
# embed_path = Path('twotower.embed')
# pairs_path = Path('./sbert-fine-tune-dataset/'+dataset+'/train.pairs') #sys.argv[1]

sbert = TwoTower(pretrained_model)
# sbert.to(device)

# if args.load_from_ori == 0:
#     if Path(model_path/pretrained_ckpt).exists():
#         # twotower.load_state_dict(torch.load(model_path/pretrained_ckpt))
#         print("load_state_dict: ", model_path/pretrained_ckpt)
#         logger.info('load_state_dict')
#         sbert.load_state_dict(torch.load(model_path/pretrained_ckpt, map_location=device))
#     # twotower.to(device)
#     sbert.to(device)



### Settings for only tuning (last 19 layers) (11+pool) ###
for name, param in list(sbert.named_parameters())[:-18]:
    # print(name)
    param.requires_grad = False
for name, param in (sbert.named_parameters()):
    if param.requires_grad == True:
        print(name)
### ####################################################### ###




# ================================================================================================================

# --- SBert ---
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
# sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# sbert = SentenceTransformer("colorfulscoop/sbert-base-ja")
# --- SBert end ---
# --- Seq Mathcer ---
from difflib import SequenceMatcher
# --- Deq End ---

def sb_multi(names):
    tmp = [sbert.encode(n) for n in names]
    return tmp

sbert_embeddings = None
ins_names = None  

class Experiment:
    def __init__(self, args):
        self.vali = False
        self.save = args.save
        self.save_prefix = "%s_%s" % (args.data_dir.split("/")[-1], args.log)

        self.hiddens = list(map(int, args.hiddens.split(",")))
        self.heads = list(map(int, args.heads.split(",")))

        self.args = args
        self.args.encoder = args.encoder.lower()
        self.args.encoder1 = args.encoder1.lower()

        self.args.decoder = args.decoder.lower()
        self.args.sampling = args.sampling
        self.args.k = int(args.k)
        self.args.margin = float(args.margin)
        self.args.alpha = float(args.alpha)

        ##ent pairs
        self.lefts_test = [i[0] for i in d.ill_test_idx]
        self.rights_test = [i[1] for i in d.ill_test_idx]

        self.lefts_train = [i[0] for i in d.ill_train_idx]
        self.rights_train = [i[1] for i in d.ill_train_idx]

        self.lefts = [i[0] for i in d.ill_idx]
        self.rights = [i[1] for i in d.ill_idx]

        if len(self.lefts) > 15000:
            self.lefts = self.lefts[len(self.lefts) - 15000:]
            self.rights = self.rights[len(self.rights) - 15000:]

        self.fc1 = torch.nn.Linear(self.hiddens[-1], self.hiddens[-1]).to(device)
        self.fc2 = torch.nn.Linear(self.hiddens[-1], self.hiddens[-1]).to(device)

        self.cached_sample = {}
        self.best_result = ()

    def evaluate(self, it, test, ins_emb, ins_emb1, mapping_emb=None, vali_flag= False):
        t_test = time.time()
        top_k = [1, 3, 5, 10, 20, 30, 50, 70, 100, 200, 300, 500, 1000]
        # print(ins_emb.shape)
        # print(len(ins_emb))
        if mapping_emb is not None:
            print("using mapping")
            left_emb = mapping_emb[test[:, 0]]
        else:
            left_emb = ins_emb[test[:, 0]]
        right_emb = ins_emb[test[:, 1]]
        distance = - sim(left_emb, right_emb, metric=self.args.test_dist, normalize=True,
                         csls_k=self.args.csls)  # normalize = True.... False can increase performance

        if self.args.two_views == 1 and self.args.fuse_embed != 1:
            left_emb1 = ins_emb1[test[:, 0]]
            right_emb1 = ins_emb1[test[:, 1]]
            distance1 = - sim(left_emb1, right_emb1, metric=self.args.test_dist, normalize=True, csls_k=self.args.csls)
            distance = distance * self.args.alp + distance1 * (1 - self.args.alp)

        if self.args.rerank:
            indices = np.argsort(np.argsort(distance, axis=1), axis=1)
            indices_ = np.argsort(np.argsort(distance.T, axis=1), axis=1)
            distance = indices + indices_.T

        tasks = div_list(np.array(range(len(test))), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            reses.append(
                pool.apply_async(multi_cal_rank, (task, distance[task, :], distance[:, task], top_k, self.args)))
        pool.close()
        pool.join()

        acc_l2r, acc_r2l = np.array([0.] * len(top_k)), np.array([0.] * len(top_k))
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
        for res in reses:
            (_acc_l2r, _mean_l2r, _mrr_l2r, _acc_r2l, _mean_r2l, _mrr_r2l) = res.get()
            acc_l2r += _acc_l2r
            mean_l2r += _mean_l2r
            mrr_l2r += _mrr_l2r
            acc_r2l += _acc_r2l
            mean_r2l += _mean_r2l
            mrr_r2l += _mrr_r2l
        mean_l2r /= len(test)
        mean_r2l /= len(test)
        mrr_l2r /= len(test)
        mrr_r2l /= len(test)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / len(test), 4)
            acc_r2l[i] = round(acc_r2l[i] / len(test), 4)

        if vali_flag is False:
            print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r.tolist(),
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
            print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l.tolist(),
                                                                                                  mean_r2l, mrr_r2l,
                                                                                                  time.time() - t_test))
        return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)

    def init_emb(self):
        print("Start Init")
        e_scale, r_scale = 1, 1
        self.ins_embeddings = nn.Embedding(d.ins_num, self.hiddens[0] * e_scale).to(device)
        self.rel_embeddings = nn.Embedding(d.rel_num, int(self.hiddens[0] * r_scale)).to(device)
        # if self.args.mytest:
        #     global sbert_embeddings
        #     global ins_names
        #     if self.args.sbert and sbert_embeddings==None:
        #         ins_names = [ d.id2ins_dict[i] for i in range(d.ins_num)]
        #         sb_embs = [sbert.encode(n[n.rindex("_")+1:]) if '_' in n else sbert.encode(n) for n in ins_names]
        #         sbert_embeddings = torch.tensor(sb_embs).to(device)
        #     elif self.args.seq and ins_names==None:
        #         ins_names = [ d.id2ins_dict[i][d.id2ins_dict[i].rindex("_")+1:] if '_' in d.id2ins_dict[i] else d.id2ins_dict[i] for i in range(d.ins_num)]

        nn.init.xavier_normal_(self.ins_embeddings.weight)
        nn.init.xavier_normal_(self.rel_embeddings.weight)

        self.enh_ins_emb = self.ins_embeddings.weight.cpu().detach().numpy()
        self.mapping_ins_emb = None
        print("Finish Init")

    def prepare_input(self, sb_fine_tune): #[For Iter only]
        graph_encoder = Encoder(self.args.encoder, self.hiddens, self.heads + [1], self.args.appkk, activation=F.elu,
                                feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2,
                                bias=False).to(device)

        knowledge_decoder = Decoder(self.args.decoder, params={
            "e_num": d.ins_num,
            "r_num": d.rel_num,
            "dim": self.hiddens[-1],
            "feat_drop": self.args.feat_drop,
            "train_dist": self.args.train_dist,
            "sampling": self.args.sampling,
            "k": self.args.k,
            "margin": self.args.margin,
            "alpha": self.args.alpha,
            "boot": self.args.bootstrap,
            # pass other useful parameters to Decoder
        }).to(device)
        # print(knowledge_decoder)x1

        train = np.array(d.ill_train_idx.tolist())
        np.random.shuffle(train)
        pos_batch = train
        print(len(pos_batch))
        # all_pos_ids = list(pos_batch.flatten('C')) ## all_pos_ids


        dataset = self.args.finetune_dataset #"KK100-JP-3epoch"
        '''
        ## SBERT model fine-tune preprocessing...
        # print("SBERT model fine-tune preprocessing...")
        logger.info('SBERT model fine-tune preprocessing...')
        dataset = self.args.finetune_dataset #"KK100-JP-3epoch"
        epoch = self.args.finetune_epoch
        print(epoch, type(epoch))
        # print("len(pos_batch) = ",len(pos_batch))

        all_pos_batch_ids = list(pos_batch.flatten('C')) ## all_pos_ids
        # all_pos_ids = [ i for i in range(d.ins_num)]
        
        global sbert_embeddings
        global ins_names
        ins_names = [ d.id2ins_dict[i] for i in range(d.ins_num)]
        all_pos_ids = [ str(i) for i in range(d.ins_num)]


        print("sb_fine_tune==1:",sb_fine_tune==1)
        print("sbert_embeddings==None:",sbert_embeddings==None)
        if sb_fine_tune == 1 or sbert_embeddings==None:
            ### [FOR ITER ONLY] ###
            ## step1. Write line-kg.idx.txt (name \t id)
    
            print("step1. Write line-kg.idx.txt (name \t id)")
            logger.info('step1. Write line-kg.idx.txt (name \t id)')
            df = pd.DataFrame()
            df['name'] = ins_names
            df['id'] = all_pos_ids
            df.to_csv("./sbert-fine-tune-dataset/"+dataset+"/line-kg.idx.txt", header=None, index=None, sep='\t')
            del df
            
            ##### Twotower 不需要此檔！！ #####
            ## step2. Write line-kg.txt (id1 " " id2 " " 1) 
            # print("step2. Write line-kg.txt (id1 " " id2 " " 1)")
            # id1_list = all_pos_batch_ids[::2 ] ## 奇數位 index 的就是 left ent 的 id
            # id2_list = all_pos_batch_ids[1::2] ## 偶數位 index 的就是 left ent 的 id
            # label_list = [ 1 for _ in range(len(id1_list))]
            # df = pd.DataFrame()
            # df['pos_left'] = id1_list
            # df['pos_right'] = id2_list
            # df['y'] = label_list
            # df.to_csv("sbert-fine-tune-dataset/line-kg.txt", header=None, index=None, sep=' ')
            # del df

            ## step3. Write train.pairs (id1 \t id2)
            print("step3. Write train.pairs (id1 \t id2)")
            logger.info('step3. Write train.pairs (id1 \t id2)')
            id1_list = all_pos_batch_ids[::2 ] ## 奇數位 index 的就是 left ent 的 id
            id2_list = all_pos_batch_ids[1::2] ## 偶數位 index 的就是 left ent 的 id
            df = pd.DataFrame()
            df['pos_left'] = id1_list
            df['pos_right'] = id2_list
            df.to_csv("./sbert-fine-tune-dataset/"+dataset+"/train.pairs", header=None, index=None, sep='\t')
            del id1_list
            del id2_list
            del df
            ### [FOR ITER ONLY END] ###
        
        print("Finish SBERT model fine-tune preprocessing...")
        logger.info('Finish SBERT model fine-tune preprocessing...')
            
        '''


        '''
        print("SBERT model training...")
        logger.info('SBERT model training...')
        # dataset = self.args.finetune_dataset #"KK100-JP-3epoch"
        # epoch = self.args.finetune_epoch

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_model ='bert-base-multilingual-cased' # 'sentence-transformers/all-mpnet-base-v2' #'bert-base-multilingual-cased' 'colorfulscoop/sbert-base-ja'
        pretrained_ckpt = str(0) #'' ##str(epoch-1) ##[For iter only]
        batch_size = 20
        model_path = Path('./sbert-fine-tune-model_status/'+dataset)
        embed_path = Path('twotower.embed')
        pairs_path = Path('./sbert-fine-tune-dataset/'+dataset+'/train.pairs') #sys.argv[1]

        sbert = TwoTower(pretrained_model)
        sbert.to(device)

        if args.load_from_ori == 0:
            if Path(model_path/pretrained_ckpt).exists():
                # twotower.load_state_dict(torch.load(model_path/pretrained_ckpt))
                print("load_state_dict: ", model_path/pretrained_ckpt)
                logger.info('load_state_dict')
                sbert.load_state_dict(torch.load(model_path/pretrained_ckpt, map_location=device))
            # twotower.to(device)
            sbert.to(device)

        

        ### Settings for only tuning (last 19 layers) (11+pool) ###
        for name, param in list(sbert.named_parameters())[:-18]:
            # print(name)
            param.requires_grad = False
        for name, param in (sbert.named_parameters()):
            if param.requires_grad == True:
                print(name)
        ### ####################################################### ###



        if not Path(f'./sbert-fine-tune-dataset/'+dataset+'/code2tokens').exists():
            dump_code2tokens(
                kg_text_path=Path('./sbert-fine-tune-dataset/'+dataset+'/line-kg.idx.txt'),
                # kg_path=Path('./sbert-fine-tune-dataset/line-kg.txt'),
                pretrained_model=pretrained_model,
                pickle_path=Path(f'./sbert-fine-tune-dataset/'+dataset+'/code2tokens'),
                device=device,
            )


        print("sb_fine_tune == 1:",sb_fine_tune == 1)
        print("sbert_embeddings==None:",sbert_embeddings==None)
        if sb_fine_tune == 1 or sbert_embeddings==None: #[For Iter only]

            ### [FOR ITER ONLY] ###
            if epoch != 0:
                print("epoch!=0 start training...")
                logger.info('epoch!=0 start training...')
                pairs_dataset = PairsDataset()
                dataloader = torch.utils.data.DataLoader(
                    pairs_dataset(
                        train_pairs_path=Path(pairs_path),
                        code2tokens_path=Path('./sbert-fine-tune-dataset/'+dataset+'/code2tokens'),
                    ),
                    batch_size=batch_size,
                    drop_last = True
                )

                # optimizer = optim.AdamW(twotower.parameters(), lr=0.00002)
                optimizer = optim.AdamW(sbert.parameters(), lr=0.00002)
                optimizer.zero_grad()
                total_loss, total_cnt = 0., 0
                for e_ in range(int(epoch)): ### Change training epochs...
                    for query_tokens, doc_tokens, labels in tqdm(dataloader):
                        # loss, scores = twotower.forward(
                        loss, scores = sbert.forward(
                            query_tokens=query_tokens,
                            doc_tokens=doc_tokens,
                            labels=labels.type(torch.float),
                        )
                        total_loss += loss
                        loss.requires_grad_(True) ### tunepool 要加這個！
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_cnt += 1

                        if (total_cnt % 10)==0:
                            logger.info(f"Avg. Loss: {total_loss/10}")
                            total_cnt, total_loss = 0., 0.
                del dataloader

            model_path.mkdir(parents=True, exist_ok=True)
            print("model_path:",model_path)
            # logger.info(f"\nsave ckpt to\t{model_path}/{int(epoch)-1}")
            logger.info(f"\nsave ckpt to\t{model_path}/{pretrained_ckpt}") ##[For iter only]
            # torch.save(twotower.state_dict(), model_path/f'{e_}')
            # torch.save(sbert.state_dict(), model_path/f'{int(epoch)-1}')
            torch.save(sbert.state_dict(), model_path/f'{pretrained_ckpt}') ##[For iter only]
            
            print("Finish SBERT model training...")
            logger.info('Finish SBERT model training...')
            ### [FOR ITER ONLY END] ### 
        '''


        print("Init SBERT embeddings:...")
        logger.info('Init SBERT embeddings:...')
        # if self.args.mytest and sbert_embeddings == None:
        print('self.args.mytest and sb_fine_tune == 1:',self.args.mytest and sb_fine_tune == 1)
        print('sbert_embeddings == None:',sbert_embeddings == None)
        if (self.args.mytest and sb_fine_tune == 1) or sbert_embeddings == None:  ## [this line is FOR ITER ONLY]
            # global sbert_embeddings
            sb_embs = []
            
            raw_dataset = RawDataset()
            dataloader_raw = torch.utils.data.DataLoader(
                raw_dataset(
                    code2tokens_path=Path('./sbert-fine-tune-dataset/'+dataset+'/code2tokens'),
                ),
                batch_size=2, #1 #100 不能設1 一定要整除?
                drop_last = True, ## 設這個才可以容許不能整除，但會丟棄最後一個不滿的batch
            )


            embeddings, dim = [], 0
            for codes, doc_tokens in tqdm(dataloader_raw):
                embeds = sbert.embeds(
                    doc_tokens=doc_tokens,
                )
                count = 0
                for code, emb in zip(codes, embeds):
                    emb_new = emb.unsqueeze(0).to(device)
                    # print(emb_new.shape) #torch.Size([1, 64, 768])
                    # print(doc_tokens['attention_mask'][count].shape) #torch.Size([1, 64])
                    sb = mean_pooling(emb_new,  doc_tokens['attention_mask'][count])
                    sentence_embeddings = F.normalize(sb, p=2, dim=1)
                    embed = ' '.join(map(str, sentence_embeddings.tolist()))
                    embeddings.append(f"{code}\t{embed}")
                    sb_embs.append(np.array(list(sentence_embeddings[0].cpu().detach().numpy())))
                    # print("count:",count)
                    count = count+1
                dim = len(embeds[0])


            # print("len(raw_dataset):",len(raw_dataset))
            ## 處理剩下被丟掉的那個 batch 的內容
            last_dict = {}
            if len(raw_dataset) % 2 != 0: 
                
                code2tokens = pickle.load(Path('./sbert-fine-tune-dataset/'+dataset+'/code2tokens').open('rb'))
                last_dict[list(code2tokens.keys())[-2]] = code2tokens[list(code2tokens.keys())[-2]]
                last_dict[list(code2tokens.keys())[-1]] = code2tokens[list(code2tokens.keys())[-1]]
                
                raw_dataset = RawDataset_last()
                dataloader_raw = torch.utils.data.DataLoader(
                    raw_dataset(
                        code2tokens_path=last_dict,# Path('code2tokens'),
                    ),
                    batch_size=2, #1 #100 不能設1 一定要整除?
                    drop_last = True, ## 設這個才可以容許不能整除，但會丟棄最後一個不滿的batch
                )
                
                for codes, doc_tokens in tqdm(dataloader_raw):
                    embeds = sbert.embeds(
                        doc_tokens=doc_tokens,
                    )
                    count = 0
                    for code, emb in zip(codes, embeds):
                        if count == 1:
                            emb_new = emb.unsqueeze(0).to(device)
                            sb = mean_pooling(emb_new,  doc_tokens['attention_mask'][count])
                            sentence_embeddings = F.normalize(sb, p=2, dim=1)
                            embed = ' '.join(map(str, sentence_embeddings.tolist()))
                            embeddings.append(f"{code}\t{embed}")
                            sb_embs.append(np.array(list(sentence_embeddings[0].cpu().detach().numpy())))
                            # embed = ' '.join(map(str, emb.tolist()))
                            # embeddings.append(f"{code}\t{embed}")
                            # sb_embs.append(np.array(list(emb.cpu().detach().numpy())))
                        count+=1
                    dim = len(embeds[0])
                    
            
            print("len(sb_embs):",len(sb_embs))
            sbert_embeddings = torch.tensor(sb_embs).to(device)
            # print(sbert_embeddings)
            # print(sbert_embeddings.shape)
            del sb_embs



        elif self.args.seq and ins_names==None:
            ins_names = [ d.id2ins_dict[i][d.id2ins_dict[i].rindex("_")+1:] if '_' in d.id2ins_dict[i] else d.id2ins_dict[i] for i in range(d.ins_num)]
        print("Finish Init SBERT embeddings...")
        logger.info('Finish Init SBERT embeddings...')





        
        neg_batch = knowledge_decoder.sampling_method(pos_batch, d.triple_idx, d.ill_train_idx,
                                                      [d.kg1_ins_ids, d.kg2_ins_ids], knowledge_decoder.k,
                                                      params={"emb": self.enh_ins_emb, "metric": self.args.test_dist})
        # print("neg_batch:\n",neg_batch)
        print(len(neg_batch))
        if self.args.two_views == 1 and self.vali is False:
            graph_encoder1 = Encoder(self.args.encoder1, self.hiddens, self.heads + [1], self.args.appkk,
                                     activation=F.elu,
                                     feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2,
                                     bias=False).to(device)
            # print(graph_encoder1)

            return graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch
        else:
            return graph_encoder, knowledge_decoder, pos_batch, neg_batch

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def get_contrastive_loss(self, enh_emb, enh_emb1, temp=0.5):
        enh_emb = self.projection(enh_emb)
        enh_emb1 = self.projection(enh_emb1)
        f = lambda x: torch.exp(x / temp)

        refl_sim = f(self.sim(enh_emb, enh_emb))
        refl_sim_sum1 = refl_sim.sum(1)
        refl_sim_diag = refl_sim.diag()
        del refl_sim

        between_sim = f(self.sim(enh_emb, enh_emb1))
        between_sim_sum1 = between_sim.sum(1)
        between_sim_diag = between_sim.diag()
        del between_sim

        loss1 = -torch.log(between_sim_diag / (between_sim_sum1 + refl_sim_sum1 - refl_sim_diag))

        refl_sim = f(self.sim(enh_emb1, enh_emb1))
        refl_sim_sum1 = refl_sim.sum(1)
        refl_sim_diag = refl_sim.diag()
        del refl_sim

        between_sim = f(self.sim(enh_emb1, enh_emb))
        between_sim_sum1 = between_sim.sum(1)
        between_sim_diag = between_sim.diag()
        del between_sim

        loss2 = -torch.log(between_sim_diag / (between_sim_sum1 + refl_sim_sum1 - refl_sim_diag))

        loss = (loss1.sum() + loss2.sum()) / (2 * len(enh_emb))

        # print(loss)
        return loss

    def get_loss(self, graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch, it):
        graph_encoder.train()
        knowledge_decoder.train()
        neg = torch.LongTensor(neg_batch).to(device)
        pos = torch.LongTensor(pos_batch).repeat(knowledge_decoder.k * 2, 1).to(device)
        use_edges = torch.LongTensor(d.ins_G_edges_idx).to(device)
        enh_emb = graph_encoder.forward(use_edges, self.ins_embeddings.weight)
        if self.args.mytest:
            if self.args.sbert:
                global sbert_embeddings
                sbert_emb = sbert_embeddings

        if self.args.two_views == 1 and self.vali is False:
            graph_encoder1.train()
            enh_emb1 = graph_encoder1.forward(use_edges, self.ins_embeddings.weight)
            enh_emb_final = enh_emb * self.args.alp + enh_emb1 * (1 - self.args.alp)
            enh_emb_final = torch.cat([enh_emb_final, sbert_emb], 1) 
            print(enh_emb_final.shape)

            # enh_emb_final = torch.cat((enh_emb, enh_emb1), dim=-1)
            if self.args.fuse_embed == 1:
                pos_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, pos)
                neg_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, neg)
                if self.args.mytest:
                    if self.args.sbert:
                        print("NEG0")
                        if not self.args.sb_w:
                            sb_neg_margin = knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")
                        elif self.args.sb_w == 'w1':
                            sb_neg_margin = neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-new':
                            sb_neg_margin = neg_score*((knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")+1)*0.5)
                        elif self.args.sb_w == 'no':
                            sb_neg_margin = 0
                        elif self.args.sb_w == 'w1-0.5':
                            sb_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-0.7':
                            sb_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-0.2':
                            sb_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w2':
                            sb_neg_margin = 1/math.e**(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")*(-5))
                    elif self.args.seq:
                        print("NEG1")
                        if not self.args.seq_w:
                            seq_neg_margin = knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq")
                        elif self.args.seq_w == 'w1':
                            seq_neg_margin = neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'no':
                            seq_neg_margin = 0
                        elif self.args.seq_w == 'w1-0.5':
                            seq_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'w1-0.7':
                            seq_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'w1-0.2':
                            seq_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))                                         
                target = torch.ones(neg_score.size()).to(device)
                if self.args.mytest:
                    if self.args.sbert:
                        loss = knowledge_decoder.loss(pos_score+sb_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                    elif self.args.seq:
                        loss = knowledge_decoder.loss(pos_score+seq_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                    else:
                        loss = knowledge_decoder.loss(pos_score, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                else:
                    loss = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha
            else:
                enh_emb = torch.cat([enh_emb, sbert_emb], 1) 
                print(enh_emb.shape)
                pos_score = knowledge_decoder.forward(enh_emb, self.rel_embeddings.weight, pos)
                neg_score = knowledge_decoder.forward(enh_emb, self.rel_embeddings.weight, neg)
                if self.args.mytest:
                    if self.args.sbert:
                        print("NEG2")
                        if not self.args.sb_w:
                            sb_neg_margin = knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")
                        elif self.args.sb_w == 'w1':
                            sb_neg_margin = neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-new':
                            sb_neg_margin = neg_score*((knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")+1)*0.5)
                        elif self.args.sb_w == 'no':
                            sb_neg_margin = 0
                        elif self.args.sb_w == 'w1-0.5':
                            sb_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-0.7':
                            sb_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-0.2':
                            sb_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w2':
                            sb_neg_margin = 1/math.e**(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")*(-5))
                    elif self.args.seq:
                        print("NEG3")
                        if not self.args.seq_w:
                            seq_neg_margin = knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq")
                        elif self.args.seq_w == 'w1':
                            seq_neg_margin = neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'no':
                            seq_neg_margin = 0
                        elif self.args.seq_w == 'w1-0.5':
                            seq_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'w1-0.7':
                            seq_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'w1-0.2':
                            seq_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))  
                target = torch.ones(neg_score.size()).to(device)
                if self.args.mytest:
                    if self.args.sbert:
                        loss = knowledge_decoder.loss(pos_score+sb_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                    elif self.args.seq:
                        loss = knowledge_decoder.loss(pos_score+seq_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                    else:
                        loss = knowledge_decoder.loss(pos_score, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                else:
                    loss = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha

                enh_emb1 = torch.cat([enh_emb1, sbert_emb], 1) 
                print(enh_emb1.shape)
                pos_score = knowledge_decoder.forward(enh_emb1, self.rel_embeddings.weight, pos)
                neg_score = knowledge_decoder.forward(enh_emb1, self.rel_embeddings.weight, neg)
                if self.args.mytest:
                    if self.args.sbert:
                        print("NEG4")
                        if not self.args.sb_w:
                            sb_neg_margin = knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")
                        elif self.args.sb_w == 'w1':
                            sb_neg_margin = neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-new':
                            sb_neg_margin = neg_score*((knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")+1)*0.5)
                        elif self.args.sb_w == 'no':
                            sb_neg_margin = 0
                        elif self.args.sb_w == 'w1-0.5':
                            sb_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-0.7':
                            sb_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w1-0.2':
                            sb_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                        elif self.args.sb_w == 'w2':
                            sb_neg_margin = 1/math.e**(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")*(-5))
                    elif self.args.seq:
                        print("NEG5")
                        if not self.args.seq_w:
                            seq_neg_margin = knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq")
                        elif self.args.seq_w == 'w1':
                            seq_neg_margin = neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'no':
                            seq_neg_margin = 0
                        elif self.args.seq_w == 'w1-0.5':
                            seq_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'w1-0.7':
                            seq_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                        elif self.args.seq_w == 'w1-0.2':
                            seq_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))  
                target = torch.ones(neg_score.size()).to(device)
                if self.args.mytest:
                    if self.args.sbert:
                        loss1 = knowledge_decoder.loss(pos_score+sb_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                    elif self.args.seq:
                        loss = knowledge_decoder.loss(pos_score+seq_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                    else:
                        loss1 = knowledge_decoder.loss(pos_score, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                else:
                    loss1 = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha

                loss = loss * self.args.alp + loss1 * (1 - self.args.alp)
                self.enh_emb = enh_emb.cpu().detach().numpy()
                self.enh_emb1 = enh_emb1.cpu().detach().numpy()

        else:
            enh_emb_final = enh_emb
            enh_emb_final = torch.cat([enh_emb_final, sbert_emb], 1) 
            print(enh_emb_final.shape)
            pos_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, pos)
            neg_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, neg)
            if self.args.mytest:
                if self.args.sbert:
                    print("NEG6")
                    if not self.args.sb_w:
                        sb_neg_margin = knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")
                    elif self.args.sb_w == 'w1':
                        sb_neg_margin = neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                    elif self.args.sb_w == 'w1-new':
                        sb_neg_margin = neg_score*((knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")+1)*0.5)
                    elif self.args.sb_w == 'no':
                        sb_neg_margin = 0
                    elif self.args.sb_w == 'w1-0.5':
                        sb_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                    elif self.args.sb_w == 'w1-0.7':
                        sb_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                    elif self.args.sb_w == 'w1-0.2':
                        sb_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos"))
                    elif self.args.sb_w == 'w2':
                        sb_neg_margin = 1/math.e**(knowledge_decoder.forward(sbert_emb, self.rel_embeddings.weight, neg, metric="cos")*(-5))
                elif self.args.seq:
                    print("NEG7")
                    if not self.args.seq_w:
                        seq_neg_margin = knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq")
                    elif self.args.seq_w == 'w1':
                        seq_neg_margin = neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                    elif self.args.seq_w == 'no':
                        seq_neg_margin = 0
                    elif self.args.seq_w == 'w1-0.5':
                        seq_neg_margin = 0.5*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                    elif self.args.seq_w == 'w1-0.7':
                        seq_neg_margin = 0.7*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq"))
                    elif self.args.seq_w == 'w1-0.2':
                        seq_neg_margin = 0.2*neg_score*(knowledge_decoder.forward(ins_names, self.rel_embeddings.weight, neg, metric="seq")) 
            target = torch.ones(neg_score.size()).to(device)
            if self.args.mytest:
                if self.args.sbert:
                    loss = knowledge_decoder.loss(pos_score+sb_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                elif self.args.seq:
                    loss = knowledge_decoder.loss(pos_score+seq_neg_margin, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
                else:
                    loss = knowledge_decoder.loss(pos_score, self.args.neg_scale*neg_score, target) * knowledge_decoder.alpha
            else:
                loss = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha

        self.enh_ins_emb = enh_emb_final.cpu().detach().numpy()  # fused embedding if two

        if self.args.two_views == 1 and self.args.contras_flag == 1 and self.vali is False:
            temperatue = 1
            left = enh_emb[self.lefts]
            left1 = enh_emb1[self.lefts]
            right = enh_emb[self.rights]
            right1 = enh_emb1[self.rights]
            loss1 = self.get_contrastive_loss(left, left1, temp=temperatue)
            loss1 += self.get_contrastive_loss(right, right1, temp=temperatue)
            # loss = loss + (0.1*loss2/2 + 0.1*loss1/2)#/2#/2
            if self.args.mytest:
                if self.args.rm_semi:
                    loss = loss1
                else:
                    loss = loss + 0.2 * loss1 / 2
            else:
                loss = loss + 0.2 * loss1 / 2  # + 0.1 * loss2/2

        return loss

    def train_and_eval(self):
        self.init_emb()
        if self.args.two_views == 1:
            graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch = self.prepare_input(sb_fine_tune=0) #[For Iter only]
            params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                      + [p for p in knowledge_decoder.parameters()]
                                      + (list(graph_encoder.parameters()))
                                      + (list(graph_encoder1.parameters()))
                                      )

            if self.args.contras_flag == 1:
                params1 = nn.ParameterList((list(self.fc1.parameters())) + (list(self.fc2.parameters())))
                opt = optim.Adam([{'params': params}, {'params': params1, 'lr': 0.00001}], lr=self.args.lr,
                                 weight_decay=0.00001)
            else:
                opt = optim.Adam(params, lr=self.args.lr, weight_decay=0.00001)  # 0.00001

        else:
            graph_encoder, knowledge_decoder, pos_batch, neg_batch = self.prepare_input(sb_fine_tune=0) #[For Iter only]
            params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                      + [p for p in knowledge_decoder.parameters()]
                                      + (list(graph_encoder.parameters()))
                                      )
            opt = optim.Adam(params, lr=self.args.lr, weight_decay=0.00001)  # 0.00001

        # print("Start training...")
        # all_neg = neg_batch #### [?????]
        for it in range(0, self.args.epoch):
            t_ = time.time()
            opt.zero_grad()
            if self.args.two_views == 1:
                loss = self.get_loss(graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch, it)
            else:
                loss = self.get_loss(graph_encoder, None, knowledge_decoder, pos_batch, neg_batch,
                                     it)

            loss.backward()
            opt.step()
            loss = loss.item()
            # for name, param in opt.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            loss_name = "loss_" + knowledge_decoder.print_name.replace("[", "_").replace("]", "_")

            if (it + 1) % self.args.update == 0:
                # logger.info("neg sampling...")
                neg_batch = knowledge_decoder.sampling_method(pos_batch, d.triple_idx, d.ill_train_idx,
                                                              [d.kg1_ins_ids, d.kg2_ins_ids], knowledge_decoder.k,
                                                              params={"emb": self.enh_ins_emb,
                                                                      "metric": self.args.test_dist, })
                # all_neg += neg_batch #### [?????]
            if self.vali is True:
                if (it + 1) % (300) == 0:
                    with torch.no_grad():
                        result = self.evaluate(it, d.ill_test_idx, self.enh_ins_emb, None, self.mapping_ins_emb, self.vali)
                        # H1 = result[0][0]
                        H1 = result[2]
                        break
            else:
                # Evaluate
                if (it + 1) % self.args.check == 0:
                    print("Start validating...")
                    with torch.no_grad():
                        if self.args.two_views == 1 and self.args.fuse_embed != 1:
                            result = self.evaluate(it, d.ill_test_idx, self.enh_emb, self.enh_emb1, self.mapping_ins_emb, self.vali)
                        else:
                            result = self.evaluate(it, d.ill_test_idx, self.enh_ins_emb, None, self.mapping_ins_emb, self.vali)
                    if it + 1 == self.args.epoch:
                        H1 = result[0][0]
                        MRR = result[2]
                        ALL_Score = result
                # self.best_result = result

        return self.enh_ins_emb, H1, MRR, ALL_Score

    def train_and_eval_val(self ,sb_fine_tune): #[For Iter only]
        self.init_emb()

        graph_encoder, knowledge_decoder, pos_batch, neg_batch = self.prepare_input(sb_fine_tune) #[For Iter only]
        if sb_fine_tune == 1:
            params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                    + [p for p in knowledge_decoder.parameters()]
                                    + (list(graph_encoder.parameters()))
                                    + (list(sbert.parameters())))
        if sb_fine_tune == 0:
            params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                  + [p for p in knowledge_decoder.parameters()]
                                  + (list(graph_encoder.parameters()))
                                  )
        opt = optim.Adam(params, lr=self.args.lr, weight_decay=0.00001)  # 0.00001

        # print("Start training...")
        for it in range(0, self.args.epoch):
            t_ = time.time()
            opt.zero_grad()
            loss = self.get_loss(graph_encoder, None, knowledge_decoder, pos_batch, neg_batch, it)

            loss.backward()
            opt.step()
            loss = loss.item()
            # for name, param in opt.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            loss_name = "loss_" + knowledge_decoder.print_name.replace("[", "_").replace("]", "_")

            if (it + 1) % self.args.update == 0:
                # logger.info("neg sampling...")
                neg_batch = knowledge_decoder.sampling_method(pos_batch, d.triple_idx, d.ill_train_idx,
                                                              [d.kg1_ins_ids, d.kg2_ins_ids], knowledge_decoder.k,
                                                              params={"emb": self.enh_ins_emb,
                                                                      "metric": self.args.test_dist, })
            if (it + 1) % (300) == 0:
                with torch.no_grad():
                    result = self.evaluate(it, d.ill_test_idx, self.enh_ins_emb, None, self.mapping_ins_emb, self.vali)
                    # H1 = result[0][0]
                    H1 = result[2]
                    break

        return self.enh_ins_emb, H1

def perc(metric_name):
    id2perc = dict()
    inf = open(args.data_dir + '/' + metric_name + '_perc.txt')
    for line in inf:
        strs = line.strip().split('\t')
        id2perc[int(strs[0])] = float(strs[1])
    return id2perc

def score(metric_name):
    id2score = dict()
    inf = open(args.data_dir + '/' + metric_name + '_1.txt')
    for line in inf:
        strs = line.strip().split('\t')
        id2score[int(strs[0])] = float(strs[1])
    return id2score

def centrality_score(lefts, ablat):
    left2score = dict()
    for item in lefts:
        if ablat == '_degree':
            metric_name = 'degree'
            ent2value_deg = perc(metric_name)
            left2score[item] = ent2value_deg[item]
        elif ablat == '_pr':
            metric_name = 'pr'
            ent2value_pr = perc(metric_name)
            left2score[item] = ent2value_pr[item]
    return left2score

def information_den(enh_emb, lefts):
    train_embed = enh_emb[lefts]
    kmeans = KMeans().fit(train_embed)
    center_embeds = kmeans.cluster_centers_
    labels = kmeans.labels_
    ent2valueD = dict()
    scores = []
    for i in range(len(lefts)):
        emb = train_embed[i]
        dis = scipy.spatial.distance.euclidean(emb, center_embeds[int(labels[i])])
        ent2valueD[lefts[i]] = 1.0 / (1 + dis)
        scores.append(1.0 / (1 + dis))
    scores.sort(reverse=True)
    score2perc = dict()
    for i in range(len(scores)):
        score2perc[scores[i]] = (len(scores) - i + 1) * 1.0 / len(scores)
    left2score = dict()
    for item in lefts:
        left2score[item] = score2perc[ent2valueD[item]] # args.theta0
    return left2score

def update_dic_perc(lefts, score_dict):
    scores = []
    for i in range(len(lefts)):
        scores.append(score_dict[lefts[i]])
    scores.sort(reverse=True)
    score2perc = dict()
    for i in range(len(scores)):
        score2perc[scores[i]] = (len(scores) - i + 1) * 1.0 / len(scores)
    left2score = dict()
    for item in lefts:
        left2score[item] = score2perc[score_dict[item]]  # args.theta0
    return left2score

def suggesting_score(lefts, score_dicts, b, U, r, num_chosen):
    suggestedEnt2Score = dict()
    suggestedEnts = []
    # obtain the weight
    weights_reward = np.zeros(3)
    weights_explore = np.zeros(3)
    alpha = 0.5
    weights = np.zeros(3)
    aaa = 0.4
    bbb = 0.2
    if r <= 5:
        weights[0] = aaa; weights[1] = aaa; weights[2] = bbb
    else:
        for kkk in range(len(score_dicts)):
            weights_reward[kkk] = np.sum(U[kkk][:r - 1])*1.0/(r-1) # all the history...
            weights_explore[kkk] = math.sqrt(1.5 * math.log(r) / num_chosen[kkk])
        if np.sum(weights_reward) == 0:
            weights_reward[0] = 0.333; weights_reward[1] = 0.333; weights_reward[2] = 0.333
        else:
            weights_reward = weights_reward/np.sum(weights_reward) # normalize
        if np.sum(weights_explore) == 0:
            weights_explore[0] = 0.333; weights_explore[1] = 0.333; weights_explore[2] = 0.333
        else:
            weights_explore = weights_explore/np.sum(weights_explore) # normalize
        weights = alpha * weights_reward + (1 - alpha) * weights_explore
    # print('weight normalize')
    # print(weights)

    for kkk in range(len(score_dicts)):
        ranks = sorted(score_dicts[kkk].items(), key=lambda d: d[1], reverse=True)
        suggested = []
        for i in range(len(lefts)):
            try:
                id = ranks[i][0]
            except:
                print(i)
                exit()
            suggested.append(id)
            if id not in suggestedEnt2Score:
                suggestedEnt2Score[id] = score_dicts[kkk][id]*weights[kkk]
            else:
                suggestedEnt2Score[id] += score_dicts[kkk][id]*weights[kkk]
        suggestedEnts.append(suggested[:b])
    return suggestedEnt2Score, suggestedEnts

def cmab(lefts, score_dicts, train_mapping, trained, ouf):
    t_total = time.time()
    N = len(score_dicts) # num of strategies
    R = 25 #40 # num of rounds
    b = 50
    U = np.ones((N, R)) # for each arm, record its u
    num_chosen = np.ones(N)
    all_chosen = []

    d.ill_train_idx = copy.deepcopy(trained)
    d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
    experiment = Experiment(args=args)
    experiment.vali = True
    _, H1 = experiment.train_and_eval_val(sb_fine_tune=0) #[For Iter only]
    H1_pre = H1*100
    # # in each iteration, each arm suggest b ents
    suggestedEnt2Score, suggestedEnts = suggesting_score(lefts, score_dicts, b, U, 1, num_chosen)
    selected = sorted(suggestedEnt2Score.items(), key=lambda d: d[1], reverse=True)[:b]
    chosen_ents = [i[0] for i in selected]
    all_chosen.extend(chosen_ents)
    lefts = list(set(lefts) - set(chosen_ents))

    #  update U for each arm, chose the overlapping part, and calculate the reward...
    # overlapping_total = [[],[],[]]
    num_chosen_this = np.zeros(N)
    for i in range(N):
        suggested = suggestedEnts[i]
        overlapping = list(set(suggested) & set(chosen_ents))
        num_chosen[i] += len(overlapping)
        num_chosen_this[i] = len(overlapping)
        new_train = []
        for item in overlapping:
            new_train.append([item, train_mapping[item]])
        if len(new_train)>0:
            d.ill_train_idx = copy.deepcopy(np.concatenate([trained, np.array(new_train)]))
            print("(A) Len of training " + str(len(d.ill_train_idx))) ####
            d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
            experiment = Experiment(args=args)
            experiment.vali = True
            _, H1 = experiment.train_and_eval_val(sb_fine_tune=0) #[For Iter only]
            H1 = H1 * 100
            gap = H1 - H1_pre
            if gap < 0:
                gap = 0.0
        else:
            gap = 0.0
        U[i][0] = gap
    # print(num_chosen)

    # remove the selected from the score_dicts
    for i in range(N):
        for ent in chosen_ents:
            del score_dicts[i][ent]
        # print(len(score_dicts[i]))

    new_train = []
    for item in chosen_ents:
        new_train.append([item, train_mapping[item]])
    trained = np.concatenate([trained, np.array(new_train)])

    d.ill_train_idx = copy.deepcopy(trained)
    d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
    experiment = Experiment(args=args)
    experiment.vali = True
    _, H1 = experiment.train_and_eval_val(args.sb_fine_tune) ## [this line is for Iter only.]
    H1 = H1 * 100
    gap = H1 - H1_pre
    if gap < 0:
        gap = 0.0
    # print("chosen gain: " +str(gap))

    num_chosen_this = num_chosen_this/50.0
    for i in range(N):
        U[i][0] += gap*num_chosen_this[i]

    # print(len(set(all_chosen)))
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    for r in range(2,R+1):
        if r<=2 and args.sb_fine_tune==1:
            sb_fine_tune_ = 1
        if r>2:
            sb_fine_tune_ = 0
        H1_pre = H1
        suggestedEnt2Score, suggestedEnts = suggesting_score(lefts, score_dicts, b, U, r, num_chosen)
        selected = sorted(suggestedEnt2Score.items(), key=lambda d: d[1], reverse=True)[:b]
        chosen_ents = [i[0] for i in selected]
        all_chosen.extend(chosen_ents)
        lefts = list(set(lefts) - set(chosen_ents))

        num_chosen_this = np.zeros(N)
        for i in range(N):
            suggested = suggestedEnts[i]
            overlapping = list(set(suggested) & set(chosen_ents))
            num_chosen[i] += len(overlapping)
            num_chosen_this[i] = len(overlapping)

            new_train = []
            for item in overlapping:
                new_train.append([item, train_mapping[item]])

            if len(new_train) > 0:
                d.ill_train_idx = copy.deepcopy(np.concatenate([trained, np.array(new_train)]))
                d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
                experiment = Experiment(args=args)
                experiment.vali = True
                _, H1 = experiment.train_and_eval_val(sb_fine_tune=0) #[For Iter only]
                H1 = H1 * 100
                gap = H1 - H1_pre
                if gap < 0:
                    gap = 0.0
            else:
                gap = 0.0
            U[i][r-1] = gap

        # print(num_chosen)
        # remove the selected from the score_dicts
        for i in range(N):
            for ent in chosen_ents:
                del score_dicts[i][ent]

        new_train = []
        for item in chosen_ents:
            new_train.append([item, train_mapping[item]])
        trained = np.concatenate([trained, np.array(new_train)])

        d.ill_train_idx = copy.deepcopy(trained)
        d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
        experiment = Experiment(args=args)
        experiment.vali = True
        _, H1 = experiment.train_and_eval_val(sb_fine_tune_) ## [this line is for Iter only.]
        H1 = H1 * 100
        gap = H1 - H1_pre
        if gap < 0:
            gap = 0.0
        # print("chosen gain: " + str(gap))
        num_chosen_this = num_chosen_this / 50.0

        for i in range(N):
            U[i][r-1] += gap * num_chosen_this[i]

        if r%5==0:
            d.ill_train_idx = copy.deepcopy(trained)
            d.ill_test_idx = copy.deepcopy(d.ill_test_idx_)
            print(str(r) + "Len of training " + str(len(d.ill_train_idx)))
            experiment = Experiment(args=args)
            enh_emb, HHHH, MRR_, ALL_Score = experiment.train_and_eval()
            all_scores_ = str(ALL_Score[0][0]) + '\t' + str(ALL_Score[0][1]) + '\t' + str(ALL_Score[0][2]) + '\t' + str(ALL_Score[0][3]) + '\t' + str(ALL_Score[0][4]) + '\t' + str(ALL_Score[0][5]) + '\t' + str(ALL_Score[0][6]) + '\t' + str(ALL_Score[0][7]) + '\t' + str(ALL_Score[0][8]) + '\t' + str(ALL_Score[0][9]) + '\t' + str(ALL_Score[0][10]) + '\t' + str(ALL_Score[0][11]) + '\t' + str(ALL_Score[0][12]) + '\t' + str(ALL_Score[1]) + '\t'  + str(ALL_Score[2]) + '\t'
            all_scores2_ = str(ALL_Score[3][0]) + '\t' + str(ALL_Score[3][1]) + '\t' + str(ALL_Score[3][2]) + '\t' + str(ALL_Score[3][3]) + '\t' + str(ALL_Score[3][4]) + '\t' + str(ALL_Score[3][5]) + '\t' + str(ALL_Score[3][6]) + '\t' + str(ALL_Score[3][7]) + '\t' + str(ALL_Score[3][8]) + '\t' + str(ALL_Score[3][9]) + '\t' + str(ALL_Score[3][10]) + '\t' + str(ALL_Score[3][11]) + '\t' + str(ALL_Score[3][12]) + '\t' + str(ALL_Score[4]) + '\t' + str(ALL_Score[5]) + '\t'
            ouf.write(str(r) + '\t'+ str(HHHH) + '\t'+ str(MRR_) + '\t' + all_scores_ + all_scores2_ + '\n')
            ouf.flush()
            # update some dicts
            score_dicts[0] = update_dic_perc(lefts, score_dicts[0])
            score_dicts[1] = update_dic_perc(lefts, score_dicts[1])
            left2score_i = information_den(enh_emb, lefts)
            score_dicts[2] = left2score_i
            assert len(score_dicts[1]) == len(score_dicts[2]) == len(score_dicts[0])

        print("Already selecting " + str(len(set(all_chosen))) + " entities...")
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))
        logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))
        print()
    return all_chosen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="", help="pre-train embedding dir (only use in transr)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epoch", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5, help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False,
                        help="whether to use early stop")  # Early stop when the Hits@1 score begins to drop on the validation sets, checked every 10 epochs.
    parser.add_argument("--share", action="store_true", default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true", default=False, help="whether to swap ill in triple")

    parser.add_argument("--bootstrap", action="store_true", default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9, help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--encoder1", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100",
                        help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0, help="dropout rate for gat layers")
    parser.add_argument("--feat_adj_dropout", type=float, default=0.2, help="feat_adj_dropout")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N", help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25", help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1",
                        help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1", help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0, help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")

    parser.add_argument("--train_dist", type=str, default="euclidean",
                        help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean",
                        help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0, help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true", default=False, help="whether to use rerank in test")

    parser.add_argument("--theta0", type=float, default=0.2, help="thres")  # 0.2
    parser.add_argument("--eta", type=float, default=0.003, help="thres")  # 0.003

    switch = 'GCNAPP_contras_active'  # APPtry1_comb_contras
    if switch == 'GCNAPP_contras_active':
        tw = 1;cf = 1;tr = False;tae = 1
    # if switch != 'GCNAPP_contras_active':
    #     tw = 0;cf = 0;tr = False;tae = 1
    parser.add_argument("--model_name", type=str, default=switch,
                        help="name of the model, GCN, GCN_active, GCNAPP, GCNAPP_active, GCNAPP_contras, GCNAPP_contras_active")
    parser.add_argument("--finetune_dataset", type=str,  default="", help="name of finetune dataset and model statue to call")
    parser.add_argument("--finetune_epoch", type=int, default=0, help="epoch of fine-tuning SBERT model")
    parser.add_argument("--sb_fine_tune", type=int, default=0, help="fine-tuning SBERT model when train_and_eval")
    parser.add_argument("--load_from_ori", type=int, default=0, help="fine-tune SBERT model from original one every round...")
    parser.add_argument("--two_views", type=int, default=tw,
                        help="whether to use two views, if not (0), the contra flag is also 0")
    parser.add_argument("--contras_flag", type=int, default=cf, help="whether to use contrastive learning")
    parser.add_argument("--train_random", action="store_true", default=tr,
                        help="random strategy for active training")  # !!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument("--train_add_embed", type=int, default=tae, help="whether to use embedding in active learning")
    
    parser.add_argument("--alp", type=float, default=0.2, help="balance two views, the first is GCN")  # 0.2
    parser.add_argument("--fuse_embed", type=int, default=1, help="fuse at the embedding level?")
    parser.add_argument("--appkk", type=int, default=5, help="fuse at the embedding level?")

    # --- My customized settings --- #
    parser.add_argument("--mytest", type=bool, default=False, help='customize settings for testing ...')
    parser.add_argument("--sbert", type=bool, default=False, help='use sbert in semi loss ...')
    parser.add_argument("--sb_w", type=str, default=None, help='weighted sbert ...')
    parser.add_argument("--rm_semi", type=bool, default=False, help='remove semi loss ...')
    parser.add_argument("--seq", type=bool, default=False, help='use Seq.matcher in semi loss ...')
    parser.add_argument("--seq_w", type=str, default=None, help='weighted Seq.mathcer ...')
    parser.add_argument("--neg_scale", type=float, default=1, help='scale to dis(u\', v\') ...')

    args = parser.parse_args()
    print(args)
    print(args.contras_flag)
    print(args.train_add_embed)
    print("args.sb_fine_tune:",args.sb_fine_tune)
    print("args.load_from_ori:",args.load_from_ori)

    logging.basicConfig(
        filename="./logs/logs_"+str(args.model_name),
        filemode='a',
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("args.sb_fine_tune:"+str(args.sb_fine_tune))
    logger.info("args.load_from_ori:"+str(args.load_from_ori))

    if args.sbert:
        if args.two_views==0:
            ouf = open('results/1en/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '_sb.txt',
               'w')
        else:
            ouf = open('results/nosw/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '_sb.txt',
                'w')
    else:
        if args.two_views==0:
            ouf = open('results/1en/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '.txt',
                'w')
        else:
            ouf = open('results/nosw/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '.txt',
               'w')

    # ouf = open('results/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '.txt',
    #            'w')

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    d = AlignmentData(data_dir=args.data_dir, rate=args.rate, share=args.share, swap=args.swap, val=args.val,
                      with_r=args.encoder.lower() == "naea")
    print(d)
    # print(d.ill_test_idx)
    # print("len(list(d.ill_test_idx)):",len(list(d.ill_test_idx)))


    ## record eval data
    if args.sbert:
        ouf1 = open('eval_data/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '_sb_eval.txt',
               'w')
    else:
        ouf1 = open('eval_data/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '_eval.txt',
               'w')
    
    eval_datas = copy.deepcopy(d.ill_test_idx)
    print("len(eval_datas):",len(eval_datas))
    for eval_pairs in eval_datas:
        # print(eval_pairs)  ##[10485 20985]...
        ouf1.write(str(eval_pairs[0]) + '\t' + str(eval_pairs[1]) + '\n')
        ouf1.flush()
    ## record eval data - END


    # if args.data_dir.split('/')[-1] == "kkv4_prime":
    #     seed_num = 100 #500
    # else:
    #     seed_num = 500

    seed_num = 100
    
    print("seed_num:", seed_num)
    seeds = d.ill_train_idx[:seed_num]
    train_active = copy.deepcopy(d.ill_train_idx[seed_num:])
    d.ill_train_idx = seeds
    trained = copy.deepcopy(seeds)

    # first round using 200 samples
    experiment = Experiment(args=args)
    t_total = time.time()
    enh_emb, H1, MRR_, ALL_Score = experiment.train_and_eval()
    all_scores_ = str(ALL_Score[0][0]) + '\t' + str(ALL_Score[0][1]) + '\t' + str(ALL_Score[0][2]) + '\t' + str(ALL_Score[0][3]) + '\t' + str(ALL_Score[0][4]) + '\t' + str(ALL_Score[0][5]) + '\t' + str(ALL_Score[0][6]) + '\t' + str(ALL_Score[0][7]) + '\t' + str(ALL_Score[0][8]) + '\t' + str(ALL_Score[0][9]) + '\t' + str(ALL_Score[0][10]) + '\t' + str(ALL_Score[0][11]) + '\t' + str(ALL_Score[0][12]) + '\t' + str(ALL_Score[1]) + '\t'  + str(ALL_Score[2]) + '\t'
    all_scores2_ = str(ALL_Score[3][0]) + '\t' + str(ALL_Score[3][1]) + '\t' + str(ALL_Score[3][2]) + '\t' + str(ALL_Score[3][3]) + '\t' + str(ALL_Score[3][4]) + '\t' + str(ALL_Score[3][5]) + '\t' + str(ALL_Score[3][6]) + '\t' + str(ALL_Score[3][7]) + '\t' + str(ALL_Score[3][8]) + '\t' + str(ALL_Score[3][9]) + '\t' + str(ALL_Score[3][10]) + '\t' + str(ALL_Score[3][11]) + '\t' + str(ALL_Score[3][12]) + '\t' + str(ALL_Score[4]) + '\t' + str(ALL_Score[5]) + '\t'
    # ouf.write(str(H1) + '\t'+ str(MRR_) + '\t' + str(ALL_Score) + '\n')
    ouf.write(str(H1) + '\t'+ str(MRR_) + '\t' + all_scores_ + all_scores2_ + '\n')
    ouf.flush()
    print("optimization finished!")
    print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))

    train_mapping = dict()
    lefts = []
    rights = []
    for item in train_active:
        train_mapping[item[0]] = item[1]
        lefts.append(item[0])
        rights.append(item[1])

    left2score_dgeree = centrality_score(lefts, '_degree')
    left2score_pr = centrality_score(lefts, '_pr')
    left2score_i = information_den(enh_emb, lefts)

    chosen = cmab(lefts, [left2score_dgeree, left2score_pr, left2score_i], train_mapping, trained, ouf)
    print()