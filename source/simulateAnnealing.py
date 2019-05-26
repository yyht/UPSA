import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import RAKE, math, random
from zpar import ZPar
from data import array_data
import torch, sys,os
import pickle as pkl
from copy import copy
from bert.bertinterface import BertEncoding, BertSimilarity
from utils import get_corpus_bleu_scores, appendtext

def output_p(sent, model):
    # list
    sent = torch.tensor(sent, dtype=torch.long).cuda()
    output = model.predict(sent) # 1,15,300003
    return output.squeeze(0).cpu().detach().numpy()

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
            sta_vec=list(np.zeros([option.num_steps-1]))
            keyword=Rake.run(line.strip())
            pos_list=tagger.tag_sentence(line.strip()).split()
            # pos=zip(*[x.split('/') for x in pos_list])[0]
            pos=list(zip(*[x.split('/') for x in pos_list]))[0]
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

def choose_action(c):
    r=np.random.random()
    c=np.array(c)
    for i in range(1, len(c)):
        c[i]=c[i]+c[i-1]
    for i in range(len(c)):
        if c[i]>=r:
            return i

def sigma_word(x):
    if x>0.7:
        return x
    elif x>0.65:
        return (x-0.65)*14
    else:
        return 0
    #return max(0, 1-((x-1))**2)
    #return (((np.abs(x)+x)*0.5-0.6)/0.4)**2

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
    if x>0.9:
        return  1-x+0.01 # 0.1-0
    elif x>0.8:
        return 1-(x-0.8)*9 # 0.1-1
    else:
        return 1
    #return max(0, 1-((x-1))**2)
    #return (((np.abs(x)+x)*0.5-0.6)/0.4)**2

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
        # debug
        # print('sss',sim_vec)
        # print(wei2)
        # sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])
        sim=min([x for x,y in zip(list(sim_vec*wei2),list(wei2)) if y>0]+[1])
        sim = sigma_word(sim)
        sims.append(sim)
    res = np.array(sims)
    return res


    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])

def similarity_keyword_batch(s1_lists, s2s, sta_vecs, id2sen, emb_word, option, model = None):
    simss= []
    for s1_list,s2, sta_vec in zip(s1_lists,s2s, sta_vecs):
        sims = similarity_keyword(s1_list, s2, sta_vec, id2sen, emb_word, option, model)
        simss.append(sims)
    return simss


    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])


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
    wei2= torch.tensor([0]+sta_vec[:emb2.size(1)-1],dtype=torch.uint8) #8
    emb_mat = torch.bmm(emb2,emb1) # K,8,7
    norm2 = 1/(torch.norm(emb2,p= 2,dim=2)+e) # K,8,8
    norm1 = 1/(torch.norm(emb1,p= 2,dim=1)+e) # K,7,7
    norm2 = torch.diag_embed(norm2) # K,15,15
    norm1 = torch.diag_embed(norm1)
    sim_mat = torch.bmm(torch.bmm(norm2, emb_mat), norm1) # K,8,7
    sim_vec,_ = torch.max(sim_mat,2)  # K,8
    sim,_ = torch.min(sim_vec[:,wei2],1)
    sim = sigma_word_bert(sim)
    return sim.cpu().numpy()

def similarity_keyword_bleu(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
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
        # debug
        # print('sss',sim_vec)
        # print(wei2)
        # sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])
        sim=min([x for x,y in zip(list(sim_vec*wei2),list(wei2)) if y>0]+[1])
        sim = sigma_word(sim)
        sims.append(sim)
    bleus = []
    for s1 in s1_list:
        actual_word_lists = [[id2sen(s2)]*len(s1_list)]
        generated_word_lists = [id2sen(s1)]
        bleu_score = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)[3]
        bleus.append(bleu_score)

    # bleus = (1.0-sigmoid(np.minimum(bleus,0.9999)))
    bleus = (1.0-np.minimum(bleus,0.99))
    res = np.array(sims)*bleus
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
    length=sequence_length[0]-1
    if mode!=2:
        ind_token=np.argsort(prob[:,: option.dict_size],1)[-search_size:] #K,100
        print(ind_token.shape)
    
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
            for i in range(0, sequence_length_new[k]-1-ind):
                input_new[: , sequence_length_new[k]-i]=input_new[: ,  sequence_length_new[k]-1-i]
        sequence_length_new=sequence_length_new+1
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
            ind_token=np.argsort(prob[: option.dict_size])[-search_size:]
            ind_token = np.concatenate([ind_token,np.array(input[0])],0)

    input_new=np.array([input[0]]*search_size)
    sequence_length_new=np.array([sequence_length[0]]*search_size)
    length=sequence_length[0]-1
    if mode==2:
        print(input_new, ind)
        for i in range(sequence_length[0]-ind-2):
            input_new[: , ind+i+1]=input_new[: , ind+i+2]
        for i in range(sequence_length[0]-1, option.num_steps-1):
            input_new[: , i]=input_new[: , i]*0+option.dict_size+1
        print(input_new, ind)
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
    return tem/tem.sum()

def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))


def samplep(probs):
    N = probs.shape[1]
    M = probs.shape[0]
    samples = []
    for i in range(M):
        a = np.random.choice(range(N), 1, replace=True, p=probs[i])
        samples.append(a[0])
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

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def data_type():
  return tf.float32

class PTBModel(object):
  #The language model.

  def __init__(self, is_training, option,is_test_LM=False):
    self._is_training = is_training
    self.batch_size = option.batch_size
    self.num_steps = option.num_steps
    size = option.hidden_size
    self.hidden_size = option.hidden_size
    self.num_layers = option.num_layers
    self.keep_prob = option.keep_prob
    vocab_size = option.vocab_size
    self._input=tf.placeholder(shape=[None, option.num_steps], dtype=tf.int32)
    self._target=tf.placeholder(shape=[None, option.num_steps], dtype=tf.int32)
    self._sequence_length=tf.placeholder(shape=[None], dtype=tf.int32)
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input)
    softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    if is_training and option.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, option.keep_prob)
    output = self._build_rnn_graph(inputs, self._sequence_length, is_training)

    output=tf.reshape(output, [-1, option.hidden_size])
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [-1, self.num_steps, vocab_size])
    self._output_prob=tf.nn.softmax(logits)
      # Use the contrib sequence loss and average over the batches
    mask=tf.sequence_mask(lengths=self._sequence_length, maxlen=self.num_steps, dtype=data_type())
    loss = tf.contrib.seq2seq.sequence_loss(
      logits,
      self._target,
      mask, 
      average_across_timesteps=True,
      average_across_batch=True)

    # Update the cost
    self._cost = loss


    #self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      option.max_grad_norm)
    optimizer = tf.train.AdamOptimizer()
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

  def _build_rnn_graph(self, inputs, sequence_length, is_training):
    return self._build_rnn_graph_lstm(inputs, sequence_length, is_training)

  def _get_lstm_cell(self, is_training):
    return tf.contrib.rnn.BasicLSTMCell(
          self.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)

  def _build_rnn_graph_lstm(self, inputs, sequence_length, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell( is_training)
      if is_training and self.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=self.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(self.num_layers)], state_is_tuple=True)
    outputs, states=tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=data_type())

    return outputs
  


def run_epoch(sess, model, input, sequence_length, target=None, mode='train'):
  #Runs the model on the given data.
  if mode=='train':
    #train language model
    _,cost = sess.run([model._train_op, model._cost], feed_dict={model._input: input, model._target:target, model._sequence_length:sequence_length})
    return cost
  elif mode=='test':
    #test language model
    cost = sess.run(model._cost, feed_dict={model._input: input, model._target:target, model._sequence_length:sequence_length})
    return cost
  else:
    #use the language model to calculate sentence probability
    output_prob = sess.run(model._output_prob, feed_dict={model._input: input, model._sequence_length:sequence_length})
    return output_prob





def metropolisHasting(option, dataclass,forwardmodel, backwardmodel):
    tfflag = True
    if tfflag:
        with tf.name_scope("forward_train"):
            with tf.variable_scope("forward", reuse=None):
                m_forward = PTBModel(is_training=True,option=option)
        with tf.name_scope("forward_test"):
            with tf.variable_scope("forward", reuse=True):
                mtest_forward = PTBModel(is_training=False,option=option)
        var=tf.trainable_variables()
        var_forward=[x for x in var if x.name.startswith('forward')]
        saver_forward=tf.train.Saver(var_forward, max_to_keep=1)

        with tf.name_scope("backward_train"):
            with tf.variable_scope("backward", reuse=None):
                m_backward = PTBModel(is_training=True,option=option)

        with tf.name_scope("backward_test"):
            with tf.variable_scope("backward", reuse=True):
                mtest_backward = PTBModel(is_training=False, option=option)
        var=tf.trainable_variables()
        var_backward=[x for x in var if x.name.startswith('backward')]
        saver_backward=tf.train.Saver(var_backward, max_to_keep=1)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        saver_forward.restore(session, option.forward_save_path)
        saver_backward.restore(session, option.backward_save_path)

    similaritymodel =  BertSimilarity()
    similarity = similarity_keyword #similarity_semantic

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    
    for sen_id in range(use_data.length):
        #generate for each sentence
        sta_vec=sta_vec_list[sen_id%len(sta_vec)]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0
        print(' '.join(id2sen(input[0])))
        print(sta_vec)

        for iter in range(option.sample_time):
            #ind is the index of the selected word, regardless of the beginning token.
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            if action==0: # word replacement (action: 0)
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                    prob_old_prob*=similarity_old
                else:
                    similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)

                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                if input_candidate[prob_candidate_ind][ind+1]<option.dict_size and\
                        (prob_candidate_prob>prob_old_prob*option.threshold or just_acc(option)==0):
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        input= input1
                        print(' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                #alpha is acceptance ratio of current proposal
                alpha=min(1, prob_candidate_prob*option.action_prob[2]/(prob_old_prob*option.action_prob[1]*prob_candidate_norm[prob_candidate_ind]))
            
                if choose_action([alpha, 1-alpha])==0 and \
                        input_candidate[prob_candidate_ind][ind]<option.dict_size and \
                        (prob_candidate_prob>prob_old_prob* option.threshold or just_acc(option)==0):
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1
                    pos+=1
                    sta_vec.insert(ind, 0.0)
                    del(sta_vec[-1])
                    print(' '.join(id2sen(input[0]))) 

            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, None, option.search_size, option, mode=action)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                # original sentence
                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=0)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=0)
                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
            
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_candidate=prob_candidate*similarity_candidate
            
                prob_candidate_norm=normalize(prob_candidate)
                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    alpha=min(prob_candidate_norm[candidate_ind]*prob_new_prob*option.action_prob[1]/(option.action_prob[2]*prob_old_prob), 1)
                else:
                    alpha=0
             
                if choose_action([alpha, 1-alpha])==0 and (prob_new_prob> prob_old_prob*option.threshold or just_acc(option)==0):
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    del(sta_vec[ind])
                    sta_vec.append(0)

                    pos -= 1
                    print(' '.join(id2sen(input[0])))

            pos += 1
        generateset.append(id2sen(input[0]))
    return generateset

def simulatedAnnealing_bat(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    tfflag = True
    print('xxxxxxxxxx')
    if tfflag:
        with tf.name_scope("forward_train"):
            with tf.variable_scope("forward", reuse=None):
                m_forward = PTBModel(is_training=True,option=option)
                print('xxxxxxxxxx')
        with tf.name_scope("forward_test"):
            with tf.variable_scope("forward", reuse=True):
                mtest_forward = PTBModel(is_training=False,option=option)
        var=tf.trainable_variables()
        var_forward=[x for x in var if x.name.startswith('forward')]
        saver_forward=tf.train.Saver(var_forward, max_to_keep=1)

        print('xxxxxxxxxx')
        with tf.name_scope("backward_train"):
            with tf.variable_scope("backward", reuse=None):
                m_backward = PTBModel(is_training=True,option=option)

        with tf.name_scope("backward_test"):
            with tf.variable_scope("backward", reuse=True):
                mtest_backward = PTBModel(is_training=False, option=option)
        var=tf.trainable_variables()
        var_backward=[x for x in var if x.name.startswith('backward')]
        saver_backward=tf.train.Saver(var_backward, max_to_keep=1)

        print('xxxxxxxxxx')
        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run()

        # saver_forward.restore(session, option.forward_save_path)
        # saver_backward.restore(session, option.backward_save_path)
        print('xxxxxxxxxx')
    generate_candidate = generate_candidate_input_batch
    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword_batch
    elif sim_mode =='keyword-bleu':
        similarity = similarity_keyword_bleu
    elif sim_mode =='keyword-bert':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert
    elif sim_mode =='keyword-bert-bleu':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert_bleu

    elif sim_mode =='semantic':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_keyword

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    C = 0.05
    batch_size = 20
    temperatures =  C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures, use_data.length)
    print(use_data.length/batch_size)
    for sen_id in range(int(use_data.length/batch_size)):
        sta_vec=sta_vec_list[sen_id*batch_size:sen_id*batch_size+batch_size]
        input, sequence_length, _=use_data(batch_size, sen_id)
        input_original=input
        N_input = len(input)
        sta_vec_original = [x for x in sta_vec]
        pos=0
        for sta, sent in zip( sta_vec, input):
            print(' '.join(id2sen(sent)))
            print(sta)
        calibrated_set = [x for x in input[0]]
        for iter in range(option.sample_time):
            temperature = temperatures[iter]
            ind=pos%(np.max(sequence_length))
            action=choose_action(option.action_prob)
            action = 0
            calibrated_set = list(set(calibrated_set))
            if action==0: # word replacement (action: 0)
                prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use') # K,L,Vocab
                prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
                input_ = [[x] for x in input]
                similarity_old=similarity(input_, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel) #K,
                V_old = prob_old_prob*np.concatenate(similarity_old,0)

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward=run_epoch(session, mtest_forward, input_forward,\
                        sequence_length_forward, mode='use')[:, ind%(sequence_length[0]-1),:]
                prob_backward=run_epoch(session, mtest_backward, input_backward,\
                        sequence_length_backward, mode='use')[:, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward) #K,vocab

                input_candidate, sequence_length_candidate=generate_candidate(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                         calibrated_set=calibrated_set) # K,100,15
                input_candidate_flat = input_candidate.reshape(-1,option.num_steps)
                sequence_length_candidate_flat = sequence_length_candidate.reshape(-1)
                prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate_flat,\
                            sequence_length_candidate_flat, mode='use') #K*100,15,vocab
                prob_candidate = getp(prob_candidate_pre,
                        input_candidate_flat,sequence_length_candidate_flat, option) # K*100

                prob_candidate = np.array(prob_candidate).reshape(N_input,-1) # K,100
                similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel) #  K,100
                similarity_candidate = np.concatenate(similarity_candidate,0).reshape(N_input,-1)
                prob_candidate=prob_candidate*similarity_candidate # K,100
                prob_candidate_norm= prob_candidate/prob_candidate.sum(1,keepdims=True)
                prob_candidate_ind=samplep(prob_candidate_norm)
                prob_candidate_prob= torch.gather(torch.tensor(prob_candidate),1,\
                        torch.tensor(prob_candidate_ind,dtype=torch.long).view(N_input,1)) # 5,1
                prob_candidate_prob = prob_candidate_prob.squeeze().numpy() 
                V_new = np.log(np.maximum(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
                V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
                for i,inp in enumerate(input):                
                    alpha = alphat[i]
                    chooseind = prob_candidate_ind[i]
                    if choose_action([alpha, 1-alpha])==0:
                        input1=input_candidate[i][chooseind]
                        if np.sum(input1)==np.sum(inp):
                            pass
                        else:
                            input[i] = input1
                            # calibrated_set.append(input[i][ind])
                            print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    continue
                    # break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                    calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                #for i in range(option.search_size):
                for i in range(len(input_candidate)):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                similarity_new = similarity_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1

                    pos+=1
                    # sta_vec.insert(ind, 0.0)
                    # del(sta_vec[-1])
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                            V_new,alphat,similarity_old,similarity_new)

                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==2: # word delete
                if sequence_length[0]<=2 or ind==0:
                    pos += 1
                    continue
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, None, option.search_size, option,\
                        mode=action,calibrated_set=calibrated_set)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
                    V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                    alphat = min(1,math.exp((V_new-V_old)/temperature))
                else:
                    alphat=0
             
                if choose_action([alphat, 1-alphat])==0:
                    calibrated_set.append(input[0][ind])
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    # del(sta_vec[ind])
                    # sta_vec.append(0)
                    pos -= 1

                    print('oldprob,vold, vnew, alpha,simold, simnew',prob_old_prob,V_old,\
                                V_new,alphat,similarity_old,similarity_candidate)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            pos += 1
        generateset.append(id2sen(input[0]))
        appendtext(id2sen(input[0]), option.save_path)
    return generateset

def simulatedAnnealing(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    tfflag = True
    with tf.name_scope("forward_train"):
        with tf.variable_scope("forward", reuse=None):
            m_forward = PTBModel(is_training=True, option=option)
    with tf.name_scope("forward_test"):
        with tf.variable_scope("forward", reuse=True):
            mtest_forward = PTBModel(is_training=False, option=option)
    var=tf.trainable_variables()
    var_forward=[x for x in var if x.name.startswith('forward')]
    saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
    with tf.name_scope("backward_train"):
        with tf.variable_scope("backward", reuse=None):
            m_backward = PTBModel(is_training=True, option=option)
    with tf.name_scope("backward_test"):
        with tf.variable_scope("backward", reuse=True):
            mtest_backward = PTBModel(is_training=False,option=option)
    var=tf.trainable_variables()
    var_backward=[x for x in var if x.name.startswith('backward')]
    saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
    print('line1295-------------------')
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)


        saver_forward.restore(session, option.forward_save_path)
        saver_backward.restore(session, option.backward_save_path)

    print('line1295-------------------')


    # if tfflag:
    #     with tf.name_scope("forward_train"):
    #         with tf.variable_scope("forward", reuse=None):
    #             m_forward = PTBModel(is_training=True,option=option)
    #     with tf.name_scope("forward_test"):
    #         with tf.variable_scope("forward", reuse=True):
    #             mtest_forward = PTBModel(is_training=False,option=option)
    #     var=tf.trainable_variables()
    #     var_forward=[x for x in var if x.name.startswith('forward')]
    #     saver_forward=tf.train.Saver(var_forward, max_to_keep=1)

    #     with tf.name_scope("backward_train"):
    #         with tf.variable_scope("backward", reuse=None):
    #             m_backward = PTBModel(is_training=True,option=option)

    #     with tf.name_scope("backward_test"):
    #         with tf.variable_scope("backward", reuse=True):
    #             mtest_backward = PTBModel(is_training=False, option=option)
    #     var=tf.trainable_variables()
    #     var_backward=[x for x in var if x.name.startswith('backward')]
    #     saver_backward=tf.train.Saver(var_backward, max_to_keep=1)

    #     init = tf.global_variables_initializer()
    #     session = tf.Session()
    #     session.run(init)

    #     saver_forward.restore(session, option.forward_save_path)
    #     saver_backward.restore(session, option.backward_save_path)

    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword
    elif sim_mode =='keyword-bleu':
        similarity = similarity_keyword_bleu
    elif sim_mode =='keyword-bert':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert
    elif sim_mode =='keyword-bert-bleu':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert_bleu

    elif sim_mode =='semantic':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_keyword

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    C = 0.05
    temperatures =  C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    
    for sen_id in range(use_data.length):
        sta_vec=sta_vec_list[sen_id]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        sta_vec_original = [x for x in sta_vec]
        # for i in range(1,option.num_steps):
        #   if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
        #     sta_vec[i-1]=1
        pos=0
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        calibrated_set = [x for x in input[0]]
        for iter in range(option.sample_time):
            temperature = temperatures[iter]
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            calibrated_set = list(set(calibrated_set))
            if action==0: # word replacement (action: 0)
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                    prob_old_prob*=similarity_old
                else:
                    similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                         calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(len(input_candidate)):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                

                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
                
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        calibrated_set.append(input[0][ind])
                        input= input1
                        print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                                        V_new,alphat,similarity_old,similarity_candidate[prob_candidate_ind])
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    continue
                    # break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                    calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                #for i in range(option.search_size):
                for i in range(len(input_candidate)):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                similarity_new = similarity_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1

                    pos+=1
                    # sta_vec.insert(ind, 0.0)
                    # del(sta_vec[-1])
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                            V_new,alphat,similarity_old,similarity_new)

                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==2: # word delete
                if sequence_length[0]<=2 or ind==0:
                    pos += 1
                    continue
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, None, option.search_size, option,\
                        mode=action,calibrated_set=calibrated_set)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
                    V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                    alphat = min(1,math.exp((V_new-V_old)/temperature))
                else:
                    alphat=0
             
                if choose_action([alphat, 1-alphat])==0:
                    calibrated_set.append(input[0][ind])
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    # del(sta_vec[ind])
                    # sta_vec.append(0)
                    pos -= 1

                    print('oldprob,vold, vnew, alpha,simold, simnew',prob_old_prob,V_old,\
                                V_new,alphat,similarity_old,similarity_candidate)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            pos += 1
        generateset.append(id2sen(input[0]))
        appendtext(id2sen(input[0]), option.save_path)
    return generateset



def simulatedAnnealing_std(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    tfflag = True
    if tfflag:
        with tf.name_scope("forward_train"):
            with tf.variable_scope("forward", reuse=None):
                m_forward = PTBModel(is_training=True,option=option)
        with tf.name_scope("forward_test"):
            with tf.variable_scope("forward", reuse=True):
                mtest_forward = PTBModel(is_training=False,option=option)
        var=tf.trainable_variables()
        var_forward=[x for x in var if x.name.startswith('forward')]
        saver_forward=tf.train.Saver(var_forward, max_to_keep=1)

        with tf.name_scope("backward_train"):
            with tf.variable_scope("backward", reuse=None):
                m_backward = PTBModel(is_training=True,option=option)

        with tf.name_scope("backward_test"):
            with tf.variable_scope("backward", reuse=True):
                mtest_backward = PTBModel(is_training=False, option=option)
        var=tf.trainable_variables()
        var_backward=[x for x in var if x.name.startswith('backward')]
        saver_backward=tf.train.Saver(var_backward, max_to_keep=1)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        saver_forward.restore(session, option.forward_save_path)
        saver_backward.restore(session, option.backward_save_path)

    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword
    elif sim_mode =='keyword-bleu':
        similarity = similarity_keyword_bleu
    elif sim_mode =='keyword-bert':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert
    elif sim_mode =='keyword-bert-bleu':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert_bleu

    elif sim_mode =='semantic':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_keyword

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    C = 0.05
    temperatures =  C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    
    for sen_id in range(use_data.length):
        sta_vec=sta_vec_list[sen_id]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        sta_vec_original = [x for x in sta_vec]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        calibrated_set = [x for x in input[0]]
        for iter in range(option.sample_time):
            temperature = temperatures[iter]
            print(temperature)
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            calibrated_set = list(set(calibrated_set))
            if action==0: # word replacement (action: 0)
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                    prob_old_prob*=similarity_old
                else:
                    similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                         calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(len(input_candidate)):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                

                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
                
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        calibrated_set.append(input[0][ind])
                        input= input1
                        print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                                        V_new,alphat,similarity_old,similarity_candidate[prob_candidate_ind])
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    continue
                    # break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                    calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                #for i in range(option.search_size):
                for i in range(len(input_candidate)):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                similarity_new = similarity_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1

                    pos+=1
                    # sta_vec.insert(ind, 0.0)
                    # del(sta_vec[-1])
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                            V_new,alphat,similarity_old,similarity_new)

                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==2: # word delete
                if sequence_length[0]<=2 or ind==0:
                    pos += 1
                    continue
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, None, option.search_size, option,\
                        mode=action,calibrated_set=calibrated_set)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
                    V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                    alphat = min(1,math.exp((V_new-V_old)/temperature))
                else:
                    alphat=0
             
                if choose_action([alphat, 1-alphat])==0:
                    calibrated_set.append(input[0][ind])
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    # del(sta_vec[ind])
                    # sta_vec.append(0)
                    pos -= 1

                    print('oldprob,vold, vnew, alpha,simold, simnew',prob_old_prob,V_old,\
                                V_new,alphat,similarity_old,similarity_candidate)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            pos += 1
        generateset.append(id2sen(input[0]))
        appendtext(id2sen(input[0]), option.save_path)
    return generateset

def simulatedAnnealing_calibrated(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    tfflag = True
    if tfflag:
        with tf.name_scope("forward_train"):
            with tf.variable_scope("forward", reuse=None):
                m_forward = PTBModel(is_training=True,option=option)
        with tf.name_scope("forward_test"):
            with tf.variable_scope("forward", reuse=True):
                mtest_forward = PTBModel(is_training=False,option=option)
        var=tf.trainable_variables()
        var_forward=[x for x in var if x.name.startswith('forward')]
        saver_forward=tf.train.Saver(var_forward, max_to_keep=1)

        with tf.name_scope("backward_train"):
            with tf.variable_scope("backward", reuse=None):
                m_backward = PTBModel(is_training=True,option=option)

        with tf.name_scope("backward_test"):
            with tf.variable_scope("backward", reuse=True):
                mtest_backward = PTBModel(is_training=False, option=option)
        var=tf.trainable_variables()
        var_backward=[x for x in var if x.name.startswith('backward')]
        saver_backward=tf.train.Saver(var_backward, max_to_keep=1)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        saver_forward.restore(session, option.forward_save_path)
        saver_backward.restore(session, option.backward_save_path)

    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword
    elif sim_mode =='keyword-bleu':
        similarity = similarity_keyword_bleu
    elif sim_mode =='keyword-bert':
        similaritymodel =  BertEncoding()
        similarity = similarity_keyword_bert
    elif sim_mode =='semantic':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_keyword

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    C = 2
    temperatures = 0.3+  C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    
    for sen_id in range(use_data.length):
        sta_vec=sta_vec_list[sen_id%len(sta_vec)]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        sta_vec_original = [x for x in sta_vec]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        calibrated_set = [x for x in input[0]]
        for iter in range(option.sample_time):
            temperature = temperatures[iter]
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            calibrated_set = list(set(calibrated_set))
            if action==0: # word replacement (action: 0)
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                    prob_old_prob*=similarity_old
                else:
                    similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                         calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(len(input_candidate)):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                V_new = math.log(max(prob_candidate_prob,1e-200))
                V_old = math.log(max(prob_old_prob,1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        calibrated_set.append(input[0][ind])
                        input= input1
                        print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                                V_new,alphat,0,0)
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                    calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                #for i in range(option.search_size):
                for i in range(len(input_candidate)):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                V_new = math.log(max(prob_candidate_prob, 1e-200))
                V_old = math.log(max(prob_old_prob*prob_candidate_norm[prob_candidate_ind],1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1

                    # debug
                    # print('xxxx', sequence_length, sta_vec)
                    # tem=1
                    # prob_old=run_epoch(session, mtest_forward, input,\
                    #         sequence_length,mode='use')[0]
                    # for j in range(sequence_length[0]-1):
                    #     tem*=prob_old[j][input[0][j+1]]
                    #     print(tem,)
                    # tem*=prob_old[j+1][option.dict_size+1]
                    # print(tem)
                    # similarity_old=similarity(input, input_original,sta_vec,\
                    #         id2sen, emb_word, option, similaritymodel)[0]
                    # print(similarity_old)


                    pos+=1
                    # sta_vec.insert(ind, 0.0)
                    # del(sta_vec[-1])
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                            V_new,alphat,0,0)

                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, None, option.search_size, option,\
                        mode=action,calibrated_set=calibrated_set)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                # original sentence
                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=0)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=0,\
                        calibrated_set=calibrated_set)
                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
            
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_candidate=prob_candidate*similarity_candidate
            
                prob_candidate_norm=normalize(prob_candidate)

                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    V_new = math.log(max(prob_new_prob*prob_candidate_norm[candidate_ind],1e-300))
                    V_old = math.log(max(prob_old_prob,1e-300))
                    alphat = min(1,math.exp((V_new-V_old)/temperature))
                else:
                    alphat=0
             
                if choose_action([alphat, 1-alphat])==0:

                    calibrated_set.append(input[0][ind])
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    # del(sta_vec[ind])
                    # sta_vec.append(0)

                    pos -= 1

                    print('oldprob,vold, vnew, alpha,simold, simnew',prob_old_prob,V_old,\
                                V_new,alphat,0,0)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            pos += 1
        generateset.append(id2sen(input[0]))
        appendtext(id2sen(input[0]), option.save_path)
    return generateset

def  simulatedAnnealing_pytorch(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    sim=option.sim
    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword
    elif sim_mode =='semantic':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertSimilarity()
        similarity = similarity_semantic_keyword

    generated_sentence = []
    fileemb = open(option.emb_path,'rb')
    emb_word,emb_id=pkl.load(fileemb, encoding = 'latin1')
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    C = 1 # 0.2
    
    for sen_id in range(use_data.length):
        #generate for each sentence
        sta_vec=sta_vec_list[sen_id%len(sta_vec)]
        sta_vec.insert(0, 0.0)
        del(sta_vec[-1])

        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0

        print('Origin Sentence:')
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        print('Paraphrase:')

        for iter in range(option.sample_time):
            #ind is the index of the selected word, regardless of the beginning token.
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            steps = float(iter/(sequence_length[0]-1))
            temperature = C/(math.log(steps+2))

            if action==0: # word replacement (action: 0)
                prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                  similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                  prob_old_prob*=similarity_old
                else:
                  similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)
                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                sim_new = similarity_candidate[prob_candidate_ind]
                sim_old =similarity_old
                V_new = math.log(max(prob_candidate_prob,1e-200))
                V_old = math.log(max(prob_old_prob,1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,10)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        input= input1
                        print('oldprob,vold, vnew,simold, simnew',prob_old_prob,V_old, V_new,sim_old, sim_new)
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))
                        action = 3

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    tem = np.power(tem,(sequence_length[0]*1.0)/(sequence_length_candidate[0]))
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,\
                            option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                #alpha is acceptance ratio of current proposal
                sim_new = similarity_candidate[prob_candidate_ind]
                sim_old =similarity_old

                V_new = math.log(max(prob_candidate_prob, 1e-200))
                V_old = math.log(max(prob_old_prob,1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size and (prob_candidate_prob>prob_old_prob* option.threshold):
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1
                    pos+=1
                    # sta_vec.insert(ind, 0.0)
                    #del(sta_vec[-1])
                    print(sta_vec)
                    print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))
                    action = 3
 

            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break

                prob_old = output_p(input, forwardmodel)
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word, \
                            option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, None, option.search_size, option, mode=action)

                # delete sentence
                prob_new = output_p(input_candidate, forwardmodel)
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]

                tem = np.power(tem,sequence_length[0]*1.0/(sequence_length_candidate[0]))
                prob_new_prob=tem
                if sim!=None:
                    similarity_new=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_new_prob=prob_new_prob*similarity_new
                
                sim_new = similarity_new[0]
                sim_old =similarity_old
                V_new = math.log(max(prob_new_prob,1e-300))
                V_old = math.log(max(prob_old_prob,1e-300))
                
                alphat = min(1,math.exp((V_new-V_old)/temperature))
                      
                if choose_action([alphat, 1-alphat])==0:
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    pos-=1
                    #del(sta_vec[ind])
                    #sta_vec.append(0)
                    
                    print(sta_vec)
                    print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))
                    action = 3

            if action==3: # word delete
                lastaction =0
                V_new1 = V_old
                V_old1 = V_new
                alphat = min(1,math.exp((V_new1-V_old1)/temperature))
                if choose_action([alphat, 1-alphat])==0:
                    if lastaction ==0:
                        print('cancel')

            pos += 1
        generated_sentence.append(id2sen(input[0]))
    return generated_sentence

def simulatedAnnealing_tem1(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    tfflag = True
    if tfflag:
        with tf.name_scope("forward_train"):
            with tf.variable_scope("forward", reuse=None):
                m_forward = PTBModel(is_training=True,option=option)
        with tf.name_scope("forward_test"):
            with tf.variable_scope("forward", reuse=True):
                mtest_forward = PTBModel(is_training=False,option=option)
        var=tf.trainable_variables()
        var_forward=[x for x in var if x.name.startswith('forward')]
        saver_forward=tf.train.Saver(var_forward, max_to_keep=1)

        with tf.name_scope("backward_train"):
            with tf.variable_scope("backward", reuse=None):
                m_backward = PTBModel(is_training=True,option=option)

        with tf.name_scope("backward_test"):
            with tf.variable_scope("backward", reuse=True):
                mtest_backward = PTBModel(is_training=False, option=option)
        var=tf.trainable_variables()
        var_backward=[x for x in var if x.name.startswith('backward')]
        saver_backward=tf.train.Saver(var_backward, max_to_keep=1)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        saver_forward.restore(session, option.forward_save_path)
        saver_backward.restore(session, option.backward_save_path)


    similaritymodel =  BertSimilarity()
    similarity = similarity_keyword #similarity_semantic

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    
    for sen_id in range(use_data.length):
        #generate for each sentence
        sta_vec=sta_vec_list[sen_id]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0
        print(' '.join(id2sen(input[0])))
        print(sta_vec)

        for iter in range(option.sample_time):
            #ind is the index of the selected word, regardless of the beginning token.
            temperature = 1
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            if action==0: # word replacement (action: 0)
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                    prob_old_prob*=similarity_old
                else:
                    similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)

                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                V_new = math.log(max(prob_candidate_prob,1e-200))
                V_old = math.log(max(prob_old_prob,1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        input= input1
                        print('oldprob,vold, vnew,simold, simnew',prob_old_prob,V_old, V_new,0,0)
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                V_new = math.log(max(prob_candidate_prob, 1e-200))
                V_old = math.log(max(prob_old_prob*prob_candidate_norm[prob_candidate_ind],1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size and (prob_candidate_prob>prob_old_prob* option.threshold):
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1
                    pos+=1
                    sta_vec.insert(ind, 0.0)
                    del(sta_vec[-1])
                    print('vold, vnew,simold, simnew',V_old, V_new,0,0)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, None, option.search_size, option, mode=action)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                # original sentence
                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=0)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=0)
                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
            
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_candidate=prob_candidate*similarity_candidate
            
                prob_candidate_norm=normalize(prob_candidate)

                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    V_new = math.log(max(prob_new_prob*prob_candidate_norm[candidate_ind],1e-300))
                    V_old = math.log(max(prob_old_prob,1e-300))
                    alphat = min(1,math.exp((V_new-V_old)/temperature))
                else:
                    alphat=0
             
                if choose_action([alphat, 1-alphat])==0 and (prob_new_prob>prob_old_prob*option.threshold):
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    del(sta_vec[ind])
                    sta_vec.append(0)

                    pos -= 1
                    print(' '.join(id2sen(input[0])))

            pos += 1
        generateset.append(id2sen(input[0]))
    return generateset


