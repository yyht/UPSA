# -*- coding: utf-8 -*-
import numpy as np
import sys, string
import os,re, RAKE
from collections import deque
from nltk.translate.bleu_score import corpus_bleu
from zpar import ZPar
from data import array_data
from copy import copy
import collections, math, torch


bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return bleu


def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()
    for i in range(len(bleu_score_weights)):
        bleu_scores[i + 1] = round(
            corpus_bleu(
                list_of_references=actual_word_lists[:len(generated_word_lists)],
                hypotheses=generated_word_lists,
                weights=bleu_score_weights[i + 1]), 4)

    return bleu_scores

def clarify(line):
    # line = re.sub(r'\d+',' sss ', line)
    printable = set(string.printable)
    line = filter(lambda x: x in printable, line)
    line = line.replace('?',' ')
    line = line.replace('(',' ( ')
    line = line.replace(')',' ) ')
    line = line.replace(']',' ')
    line = line.replace('[',' ')
    line = line.replace('{',' ')
    line = line.replace('}',' ')
    line = line.replace('-',' ')
    line = line.replace('!',' ! ')
    line = line.replace('.',' . ')
    line = line.replace(',',' , ')
    line = line.replace(';',' ; ')
    line = line.replace('\'',' \' ')
    line = line.replace('\' s',' \'s')
    line = line.replace('\' t',' \'t')
    line = line.replace('"',' ')

    return line

def savetexts(sent_list, file_name):
    # list(list(word))
    fileobject = open(file_name, 'w')
    for sent in sent_list:
        fileobject.write(' '.join(sent))
        fileobject.write('\n')
    fileobject.close()

def appendtext(text, file_name):
    # list(list(word))
    fileobject = open(file_name, 'a+')
    fileobject.write(text)
    fileobject.write('\n')
    fileobject.close()


def keyword_pos2sta_vec(option,keyword, pos):
    key_ind=[]
    # pos=pos[:option.num_steps-1]
    pos=pos[:option.num_steps-1]
    for i in range(len(pos)):
        if pos[i]=='NNP':
            key_ind.append(i)
        elif pos[i] in ['NN', 'NNS'] and keyword[i]==1:
            key_ind.append(i)
        elif pos[i] in ['VBZ'] and keyword[i]==1:
            key_ind.append(i)
        elif keyword[i]==1:
            key_ind.append(i)
        elif pos[i] in ['NN', 'NNS','VBZ']:
            key_ind.append(i)
    key_ind=key_ind[:max(int(option.max_key_rate*len(pos)), option.max_key)]
    sta_vec=[]
    for i in range(len(keyword)):
        if i in key_ind:
            sta_vec.append(1)
        else:
            sta_vec.append(0)
    # liuxg
    if np.sum(sta_vec)==0:
        sta_vec[0] =1
    return sta_vec

def read_data_use(option,  sen2id):

    file_name = option.use_data_path
    max_length = option.num_steps
    dict_size = option.dict_size
    Rake = RAKE.Rake(RAKE.SmartStopList())
    z=ZPar(option.pos_path)
    tagger = z.get_tagger()
    with open(file_name) as f:
        data=[]
        vector=[]
        sta_vec_list=[]
        j=0
        for line in f:
            if len(line.strip().split())>15:
                line = ' '.join( line.strip().split()[:15])
            sta_vec=list(np.zeros([option.num_steps-1]))
            keyword=Rake.run(line.strip())
            pos_list=tagger.tag_sentence(line.strip()).split()
            pos=zip(*[x.split('/') for x in pos_list])[0]
            #pos=list(zip(*[x.split('/') for x in pos_list]))[0]
            if keyword!=[]:
                keyword=list(list(zip(*keyword))[0])
                keyword_new=[]
                linewords = line.strip().split()
                for i in range(len(linewords)):
                    for item in keyword:
                        length11 = len(item.split())
                        if ' '.join(linewords[i:i+length11])==item:
                            keyword_new.extend([i+k for k in range(length11)])
                for i in range(len(keyword_new)):
                    ind=keyword_new[i]
                    if ind<=option.num_steps-2:
                        sta_vec[ind]=1
            if option.keyword_pos==True:
                sta_vec_list.append(keyword_pos2sta_vec(option,sta_vec,pos))
            else:
                sta_vec_list.append(list(np.zeros([option.num_steps-1])))
            data.append(sen2id(line.strip().lower().split()))
    data_new=array_data(data, max_length, dict_size)
    return data_new, sta_vec_list # sentence, keyvector

def read_data_use1(option,  sen2id):

    file_name = option.use_data_path
    max_length = option.num_steps
    dict_size = option.dict_size
    Rake = RAKE.Rake(RAKE.SmartStopList())
    z=ZPar(option.pos_path)
    tagger = z.get_tagger()
    with open(file_name) as f:
        data=[]
        vector=[]
        sta_vec_list=[]
        j=0
        for line in f:
            print('sentence:'+line)
            sta_vec=list(np.zeros([option.num_steps-1]))
            keyword=Rake.run(line.strip())
            pos_list=tagger.tag_sentence(line.strip()).split()
            # pos=zip(*[x.split('/') for x in pos_list])[0]
            pos=list(zip(*[x.split('/') for x in pos_list]))[0]
            print(keyword)
            if keyword!=[]:
                keyword=list(list(zip(*keyword))[0])
                keyword_new=[]
                for item in keyword:
                    tem1=[line.strip().split().index(x) for x in item.split() if x in line.strip().split()]
                    print('id',tem1)
                    keyword_new.extend(tem1)
                print(keyword_new)
                for i in range(len(keyword_new)):
                    ind=keyword_new[i]
                    if ind<=option.num_steps-2:
                        sta_vec[ind]=1
            if option.keyword_pos==True:
                sta_vec_list.append(keyword_pos2sta_vec(option,sta_vec,pos))
            else:
                sta_vec_list.append(list(np.zeros([option.num_steps-1])))
            print(keyword_pos2sta_vec(option,sta_vec, pos))        
            data.append(sen2id(line.strip().lower().split()))
    data_new=array_data(data, max_length, dict_size)
    return data_new, sta_vec_list # sentence, keyvector

def sigma_word(x):
    if x>0.7:
        return x
    elif x>0.65:
        return (x-0.65)*14
    else:
        return 0
    #return max(0, 1-((x-1))**2)
    #return (((np.abs(x)+x)*0.5-0.6)/0.4)**2

def sigma_word_batch(x):
    # x:K,
    x9 = torch.gt(x,0.7).float()
    x8 = (torch.gt(x,0.65)*torch.lt(x, 0.7)).float()
    return 6*(x*x9+(x-0.65)*14*x8)

def sigma_word_batch1(x):
    # x:K,
    x9 = torch.gt(x,0.7).float()
    x8 = torch.gt(x,0.65).float()
    return x*x9+(x-0.65)*14*x8


def sigma_word1(x):
    if x>0.9:
        return x
    elif x>0.8:
        return (x-0.8)*9
    else:
        return 0
    #return max(0, 1-((x-1))**2)
    #return (((np.abs(x)+x)*0.5-0.6)/0.4)**2

def sigma_word_bert(x):
    # x:K,
    x9 = torch.gt(x,0.9).float()
    x8 = torch.gt(x,0.8).float()
    return x*x9+(x-0.8)*9*x8

def sigma_bleu(x):
    # if x>0.8:
    #     return  1-x+0.01 # 0.2-0
    # elif x>0.4:
    #     return 1-(x-0.4)*2 # 0.2-1
    # else:
    #     return 1
    return 1-x+0.01

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sen2mat(s, id2sen, emb_word, option):
    mat=[]
    for item in s:
        if item==option.dict_size+2:
            continue
        if item==option.dict_size+1:
            break
        word=id2sen([item])[0]
        if  word in emb_word:
            mat.append(np.array(emb_word[word]))
        else:
            mat.append(np.random.random([option.hidden_size]))
    return np.array(mat)

def similarity_semantic(s1_list,s2, sta_vec, id2sen, emb_word, option, model):
    K = 4
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    sourcesent2 = [' '.join(id2sen(s2))] * len(s1_list)
    rep1 = model.get_encoding(sourcesent, sourcesent)
    rep2 = model.get_encoding(sourcesent,sourcesent2)
    rep3 = model.get_encoding(sourcesent2,sourcesent2)
    rep1 = (rep1+rep3)/2
    norm1 = rep1.norm(2,1)
    norm2 = rep2.norm(2,1)
    semantic = torch.sum(rep1*rep2,1)/(norm1*norm2)
    semantic = semantic*(1- (torch.abs(norm1-norm2)/torch.max(norm1,norm2)))
    semantics = semantic.cpu().numpy()
    res = np.power(semantics,K)
    return res

def similarity_semantic_bleu(s1_list,s2, sta_vec, id2sen, emb_word, option, model):
    K = 12
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    sourcesent2 = [' '.join(id2sen(s2))] * len(s1_list)
    rep1 = model.get_encoding(sourcesent, sourcesent)
    rep2 = model.get_encoding(sourcesent,sourcesent2)
    rep3 = model.get_encoding(sourcesent2,sourcesent2)
    rep1 = (rep1+rep3)/2
    norm1 = rep1.norm(2,1)
    norm2 = rep2.norm(2,1)
    semantic = torch.sum(rep1*rep2,1)/(norm1*norm2)
    semantic = semantic*(1- (torch.abs(norm1-norm2)/torch.max(norm1,norm2)))
    semantics = semantic.cpu().numpy()
    bleus = []
    for s1 in s1_list:
        actual_word_lists = [[id2sen(s2)]*len(s1_list)]
        generated_word_lists = [id2sen(s1)]
        bleu_score = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)[1]
        bleus.append(bleu_score)

    bleus = (1.0-sigmoid(np.minimum(bleus,0.999)))
    semantics = np.power(semantics,K)
    res = bleus*semantics
    return res

def similarity_semantic_keyword(s1_list,s2, sta_vec, id2sen, emb_word, option, model):
    C1 = 0.1
    K = 4
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    sourcesent2 = [' '.join(id2sen(s2))] * len(s1_list)
    rep1 = model.get_encoding(sourcesent, sourcesent)
    rep2 = model.get_encoding(sourcesent,sourcesent2)
    rep3 = model.get_encoding(sourcesent2,sourcesent2)
    rep1 = (rep1+rep3)/2
    norm1 = rep1.norm(2,1)
    norm2 = rep2.norm(2,1)
    semantic = torch.sum(rep1*rep2,1)/(norm1*norm2)
    semantic = semantic*(1- (torch.abs(norm1-norm2)/torch.max(norm1,norm2)))
    semantics = semantic.cpu().numpy()
    res = np.power(semantics,K)
    semantics = []
    for s, s1 in zip(res, s1_list):
        tem = 1
        for i,x in zip(sta_vec,s2):
            if i==1 and x not in s1:
                tem *= C1
        semantics.append(s*tem)
    res = np.array(semantics)
    return res

def similarity_keyword(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    sims=  []
    for s1 in s1_list:
        emb1=sen2mat(s1, id2sen, emb_word, option) # M*K
        #wei2=normalize( np.array([-np.log(id2freq[x]) for x in s2 if x<=config.dict_size]))
        emb2=sen2mat(s2, id2sen, emb_word, option) # N*k
        wei2=np.array(sta_vec[:len(emb2)]).astype(np.float32) # N*1
        #wei2=normalize(wei2)
        
        emb_mat=np.dot(emb2,emb1.T) #N*M
        norm1=np.diag(1/(np.linalg.norm(emb1,2,axis=1)+e)) # M*M
        norm2=np.diag(1/(np.linalg.norm(emb2,2,axis=1)+e)) #N*N
        sim_mat=np.dot(norm2,emb_mat).dot(norm1) #N*M
        sim_vec=sim_mat.max(axis=1) #N
        sim=min([x for x,y in zip(list(sim_vec*wei2),list(wei2)) if y>0]+[1])
        sim = sigma_word(sim)
        sims.append(sim)
        # print(' '.join(id2sen(s1)), ' '.join(id2sen(s2)), sim)
        # print(sim_vec*wei2)
    res = np.array(sims)
    return res

    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])

def similarity_batch(s1_lists, s2s, sta_vecs, id2sen, emb_word, option, simfun, model = None):
    simss= []
    for s1_list,s2, sta_vec in zip(s1_lists,s2s, sta_vecs):
        sims = simfun(s1_list, s2, sta_vec, id2sen, emb_word, option, model)
        simss.append(sims)
    res = np.concatenate(simss,0)
    return res

def similarity_keyword_tensor(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    N_candidant = len(s1_list) 
    sims=  []
    embs = []
    for s1 in s1_list:
        emb1=sen2mat(s1, id2sen, emb_word, option) # M*K
        embs.append(np.expand_dims(emb1,axis=0))
    emb1 = np.concatenate(embs,0) # K,8,300
    emb1 = torch.tensor(emb1, dtype=torch.float).permute(0,2,1).cuda()
    emb2= sen2mat(s2, id2sen, emb_word, option) # N*k
    emb2 = torch.tensor(emb2, dtype=torch.float).unsqueeze(0).repeat(N_candidant,1,1).cuda()
    # print(emb1.size(), emb2.size()) #bs,300,7, bs,8,300
    wei2= torch.tensor(sta_vec[:emb2.size(1)],dtype=torch.uint8) #8
    emb_mat = torch.bmm(emb2,emb1) # K,8,7
    norm2 = 1/(torch.norm(emb2,p= 2,dim=2)+e) # K,8,8
    norm1 = 1/(torch.norm(emb1,p= 2,dim=1)+e) # K,7,7
    norm2 = torch.diag_embed(norm2) # K,15,15
    norm1 = torch.diag_embed(norm1)
    sim_mat = torch.bmm(torch.bmm(norm2, emb_mat), norm1) # K,8,7
    sim_vec,_ = torch.max(sim_mat,2)  # K,8
    sim,_ = torch.min(sim_vec[:,wei2],1)
    sim = sigma_word_batch(sim)
    return sim.cpu().numpy()


def similarity_keyword_bleu_tensor(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    M_kw = option.M_kw
    M_bleu = option.M_bleu
    N_candidant = len(s1_list) 
    sims=  []
    embs = []
    bleus = []
    for s1 in s1_list:
        emb1=sen2mat(s1, id2sen, emb_word, option) # M*K
        embs.append(np.expand_dims(emb1,axis=0))
        if len(id2sen(s1))==0:
            return np.array([0])

    emb1 = np.concatenate(embs,0) # K,8,300
    emb2= sen2mat(s2, id2sen, emb_word, option) # N*k
    emb1 = torch.tensor(emb1, dtype=torch.float).permute(0,2,1)
    emb2 = torch.tensor(emb2, dtype=torch.float).unsqueeze(0).repeat(N_candidant,1,1)
    wei2= torch.tensor(sta_vec[:emb2.size(1)],dtype=torch.uint8) #8
    # print(emb1.size(), emb2.size(), wei2.size()) #bs,300,7, bs,8,300
    emb_mat = torch.bmm(emb2,emb1) # K,8,7
    norm2 = 1/(torch.norm(emb2,p= 2,dim=2)+e) # K,8,8
    norm1 = 1/(torch.norm(emb1,p= 2,dim=1)+e) # K,7,7
    norm2 = torch.diag_embed(norm2) # K,15,15
    norm1 = torch.diag_embed(norm1)
    sim_mat = torch.bmm(torch.bmm(norm2, emb_mat), norm1) # K,8,7
    sim_vec,_ = torch.max(sim_mat,2)  # K,8
    sim,_ = torch.min(sim_vec[:,wei2],1)
    sim = np.power(sim.numpy(), M_kw)

    for s1 in s1_list:
        actual_word_lists = [[id2sen(s2)]]
        generated_word_lists = [id2sen(s1)]
        bleu_score = compute_bleu(actual_word_lists, generated_word_lists)
        bleu_score = 1-bleu_score+0.01
        bleus.append(bleu_score)
    bleus = np.power(np.array(bleus),M_bleu)
    res = sim*bleus
    return res



def similarity_keyword_bleu(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    sims=  []
    bleus = []
    for s1 in s1_list:
        emb1=sen2mat(s1, id2sen, emb_word, option) # M*K
        #wei2=normalize( np.array([-np.log(id2freq[x]) for x in s2 if x<=config.dict_size]))
        emb2=sen2mat(s2, id2sen, emb_word, option) # N*k
        wei2=np.array(sta_vec[:len(emb2)]).astype(np.float32) # N*1
        #wei2=normalize(wei2)
        
        emb_mat=np.dot(emb2,emb1.T) #N*M
        norm1=np.diag(1/(np.linalg.norm(emb1,2,axis=1)+e)) # M*M
        norm2=np.diag(1/(np.linalg.norm(emb2,2,axis=1)+e)) #N*N
        sim_mat=np.dot(norm2,emb_mat).dot(norm1) #N*M
        sim_vec=sim_mat.max(axis=1) #N
        # debug
        # print('sss',sim_vec)
        # print(wei2)
        # sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])
        sim=min([x for x,y in zip(list(sim_vec*wei2),list(wei2)) if y>0]+[1])
        sim = sigma_word(sim)
        sims.append(sim)

        actual_word_lists = [[id2sen(s2)]]
        generated_word_lists = [id2sen(s1)]
        # bleu_score = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)[3]
        bleu_score = compute_bleu(actual_word_lists, generated_word_lists)
 #       bleu_score = sigma_bleu(bleu_score)
        bleu_score = 1-bleu_score+0.01
        bleus.append(bleu_score)
    M_bleu = 1
    M_kw = 3
    bleus = np.power(np.array(bleus),M_bleu)
 
    sim = np.power(np.array(sim), M_kw)
    # bleus = (1.0-sigmoid(np.minimum(bleus,0.9999)))
    res = bleus*sim
    return res

def similarity_keyword_bert(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    sims=  []
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    sourcesent2 = [' '.join(id2sen(s2))]
    sourcesent = sourcesent+sourcesent2
    emb = model.get_representation(sourcesent)
    N_candidant = len(s1_list) 
    emb2 = emb[-1,:,:].unsqueeze(0).repeat(N_candidant,1,1) # K,15*d
    emb1 = emb[:-1,:,:].permute(0,2,1) #K,d,15

    wei2= torch.tensor([0]+sta_vec,dtype=torch.uint8)
    emb_mat = torch.bmm(emb2,emb1) # K,15,15
    norm2 = 1/(torch.norm(emb2,p= 2,dim=2)+e) # K,15
    norm1 = 1/(torch.norm(emb1,p= 2,dim=1)+e) # K,15
    norm2 = torch.diag_embed(norm2) # K,15,15
    norm1 = torch.diag_embed(norm1)
    sim_mat = torch.bmm(torch.bmm(norm2, emb_mat), norm1) # K,15,15
    sim_vec,_ = torch.max(sim_mat,2)  # K,15
    sim,_ = torch.min(sim_vec[:,wei2],1)
    sim = sigma_word_bert(sim)
    return sim.cpu().numpy()


    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])

def similarity_keyword_bert_bleu(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    sims=  []
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    sourcesent2 = [' '.join(id2sen(s2))]
    sourcesent = sourcesent+sourcesent2
    emb = model.get_representation(sourcesent).numpy()
    
    emb2 = emb[-1,:,:]
    actual_word_lists = [[id2sen(s2)]]
    bleus = [] 
    for i,s1 in enumerate(s1_list):
        emb1 = emb[i,:,:]
        wei2=np.array([0]+sta_vec).astype(np.float32) # N*1
        #wei2=normalize(wei2)
        
        emb_mat=np.dot(emb2,emb1.T) #N*M
        norm1=np.diag(1/(np.linalg.norm(emb1,2,axis=1)+e)) # M*M
        norm2=np.diag(1/(np.linalg.norm(emb2,2,axis=1)+e)) #N*N
        sim_mat=np.dot(norm2,emb_mat).dot(norm1) #N*M
        sim_vec=sim_mat.max(axis=1) #N
        # debug
        # print('sss',sim_vec)
        # print(wei2)
        # sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])
        sim=min([x for x,y in zip(list(sim_vec*wei2),list(wei2)) if y>0]+[1])
        sim = sigma_word1(sim)
        sims.append(sim)

        generated_word_lists = [id2sen(s1)]
        bleu_score = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)[3]
        bleu_score = sigma_bleu(bleu_score)
        bleus.append(bleu_score)
        
    # bleus = (1.0-sigmoid(np.minimum(bleus,0.9999)))
    res = np.array(sims)*np.array(bleus)
    return res


    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])

def cut_from_point(input, sequence_length, ind,option, mode=0):
    batch_size=input.shape[0]
    num_steps=input.shape[1]
    input_forward=np.zeros([batch_size, num_steps])+option.dict_size+1
    input_backward=np.zeros([batch_size, num_steps])+option.dict_size+1
    sequence_length_forward=np.zeros([batch_size])
    sequence_length_backward=np.zeros([batch_size])
    for i in range(batch_size):
        input_forward[i][0]=option.dict_size+2
        input_backward[i][0]=option.dict_size+2
        length=sequence_length[i]-1

        for j in range(ind):
            input_forward[i][j+1]=input[i][j+1]
        sequence_length_forward[i]=ind+1
        if mode==0:
            for j in range(length-ind-1):
                input_backward[i][j+1]=input[i][length-j]
            sequence_length_backward[i]=length-ind
        elif mode==1:
            for j in range(length-ind):
                input_backward[i][j+1]=input[i][length-j]
            sequence_length_backward[i]=length-ind+1
    return input_forward.astype(np.int32), input_backward.astype(np.int32), sequence_length_forward.astype(np.int32), sequence_length_backward.astype(np.int32)
   
def generate_candidate_input(input, sequence_length, ind, prob, search_size, option, mode=0):
    input_new=np.array([input[0]]*search_size)
    sequence_length_new=np.array([sequence_length[0]]*search_size)
    length=sequence_length[0]-1
    if mode!=2:
        ind_token=np.argsort(prob[: option.dict_size])[-search_size:]
    
    if mode==2:
        for i in range(sequence_length[0]-ind-2):
            input_new[: , ind+i+1]=input_new[: , ind+i+2]
        for i in range(sequence_length[0]-1, option.num_steps-1):
            input_new[: , i]=input_new[: , i]*0+option.dict_size+1
        sequence_length_new=sequence_length_new-1
        return input_new[:1], sequence_length_new[:1]
    if mode==1:
        for i in range(0, sequence_length_new[0]-1-ind):
            input_new[: , sequence_length_new[0]-i]=input_new[: ,  sequence_length_new[0]-1-i]
        sequence_length_new=sequence_length_new+1
    for i in range(search_size):
        input_new[i][ind+1]=ind_token[i]
    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)
  
def generate_candidate_input_batch(input, sequence_length, ind, prob, search_size, option, mode=0,\
        calibrated_set=None):
    # input, K,L; prob, K,vocab
    input_new=np.array([[inp]*search_size for inp in input]) # K,100,L
    sequence_length_new=np.array([[length]*search_size for length in sequence_length]) #K,100
    if mode!=2:
        ind_token=np.argsort(prob[:,: option.dict_size],1) #K,vocab
        ind_token = ind_token[:,-search_size:] #K,100
    
    if mode==2:
        for k in range(len(input)):
            for i in range(sequence_length[k]-ind-2):
                input_new[k,: , ind+i+1]=input_new[k,: , ind+i+2]
            for i in range(sequence_length[k]-1, option.num_steps-1):
                input_new[k,: , i]=input_new[k,:, i]*0+option.dict_size+1
        sequence_length_new=sequence_length_new-1
        return input_new, sequence_length_new
    if mode==1:
        for k in range(len(input)):
            tem_len = min(sequence_length[k],14) # avoid overflow
            for i in range(0, tem_len-1-ind):
                input_new[k ,:, tem_len-i]=input_new[k ,:,  tem_len-1-i]
            sequence_length_new[k,:]= np.minimum(sequence_length_new[k,:]+1,15)

    for i in range(search_size):
        input_new[:,i,ind+1]=ind_token[:,i]
    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)

def generate_candidate_input_calibrated(input, sequence_length, ind, prob, searching_size, option,\
        mode=0, calibrated_set = None):
    search_size = searching_size
    if mode!=2:
        if calibrated_set is None:
            ind_token=np.argsort(prob[: option.dict_size])[-search_size:]
        else:
            search_size = searching_size+len(calibrated_set)
            ind_token=np.argsort(prob[: option.dict_size])[-searching_size:]
            ind_token = np.concatenate([ind_token,np.array(calibrated_set)],0)

    input_new=np.array([input[0]]*search_size)
    sequence_length_new=np.array([sequence_length[0]]*search_size)
    length=sequence_length[0]-1
    if mode==2:
        for i in range(sequence_length[0]-ind-2):
            input_new[: , ind+i+1]=input_new[: , ind+i+2]
        for i in range(sequence_length[0]-1, option.num_steps):
            input_new[: , i]=input_new[: , i]*0+option.dict_size+1
        sequence_length_new=sequence_length_new-1
        return input_new[:1], sequence_length_new[:1]
    if mode==1:
        for i in range(0, sequence_length_new[0]-1-ind):
            input_new[: , sequence_length_new[0]-i]=input_new[: ,  sequence_length_new[0]-1-i]
        sequence_length_new=sequence_length_new+1
    for i in range(search_size):
        input_new[i][ind+1]=ind_token[i]
    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)

def normalize(x, e=0.05):
    tem = copy(x)
    if max(tem)==0:
        tem += e
    return tem/tem.sum()

def choose_action(c):
    r=np.random.random()
    c=np.array(c)
    for i in range(1, len(c)):
        c[i]=c[i]+c[i-1]
    for i in range(len(c)):
        if c[i]>=r:
            return i

def choose_an_action(c):
    # c,list
    r=np.random.random()
    if np.sum(c)==0:
        return len(c)-1
    for i in range(len(c)):
        if i>0:
            c[i]=c[i]+c[i-1]
        if c[i]>=r:
            return i
    return len(c)-1



def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))


def samplep(probs):
    N = probs.shape[1]
    M = probs.shape[0]
    samples = []
    for i in range(M):
        a = choose_an_action(probs[i])
        samples.append(a)
    return np.array(samples)

def just_acc(option):
    r=np.random.random()
    if r<option.just_acc_rate:
        return 0
    else:
        return 1

def getp(probabilities,input, lengths, option):
    tems = []
    for probs,inp, length in zip(probabilities,input,lengths):
        tem = 1
        for i in range(length-1):
            tem*= probs[i][inp[i+1]]
        tem*= probs[length-1][option.dict_size+1]
        tems.append(tem)
    return tems

def getppl(probabilities,input, lengths, option):
    tems = []
    for probs,inp, length in zip(probabilities,input,lengths):
        tem = 1
        for i in range(length-1):
            tem*= probs[i][inp[i+1]]
        tem*= probs[length-1][option.dict_size+1]
        tems.append(np.power(tem,1.0/length))
    return tems


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()





class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

if __name__ == "__main__":
    sent = 'I 地方have999 a33)) pretty-computer.'
    print(clarify(sent))
