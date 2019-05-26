import numpy as np 
import pickle as pkl
import os, random
import copy
from math import ceil
from collections import Counter
import torch

class Dicts(object):
    def __init__(self, dict_path):
        f = open(dict_path,'rb')
        self.Dict1, self.Dict2=pkl.load(f)
        f.close()
        self.vocab_size=len(self.Dict1)+3
        self.UNK=self.vocab_size-3
        self.BOS=self.vocab_size-1
        self.EOS=self.vocab_size-2
        
    def sen2id(self, s):
        if s==[]:
            return []
        Dict=self.Dict1
        dict_size=len(Dict)
        s_new=[]
        if type(s[0])!=type([]):
            for item in s:
                if item in Dict:
                    s_new.append(Dict[item])
                else:
                    s_new.append(dict_size)
            return s_new
        else:
            return [self.sen2id(x) for x in s]
    
    def id2sen(self, s):
        if s==[]:
            return []
        Dict=self.Dict2
        dict_size=len(Dict)
        s_new=[]
        if type(s[0])!=type([]):
            for item in s:
                if item in Dict:
                    s_new.append(Dict[item])
                elif item==dict_size:
                    s_new.append( 'UNK')
                else:
                    pass
            return s_new
        else:
            return [self.id2sen(x) for x in s]

class Data(object):
    def __init__(self, option):
        self.option = option
        dict_use = Dicts(option.dict_path)
        self.sen2id=dict_use.sen2id
        self.id2sen=dict_use.id2sen
        self.train_data, self.valid_data, self.test_data = self.read_data(self.option.data_path,\
                self.option.num_steps)
        self.test_start  = 0
        self.train_start = 0
        self.valid_start = 0

    def read_data(self,file_name,  max_length):
        dict_size = self.option.dict_size
        tt_proportion = 0.9
        if file_name[-3:]=='pkl':
            data=pkl.load(open(file_name))
        else:
            with open(file_name) as f:
                data=[]
                for line in f:
                    line = line.strip().lower()
                    data.append(self.sen2id(line.split()))
        train_data=array_data(data[ : int(len(data)*(tt_proportion-0.05))], max_length, dict_size, shuffle=True)
        valid_data=array_data(data[int(len(data)*(tt_proportion-0.05)): int(len(data)*tt_proportion)], max_length, dict_size, shuffle=True)
        test_data=array_data(data[int(len(data)*tt_proportion) : ], max_length, dict_size, shuffle=True)
        return train_data, valid_data, test_data

    def _next_batch(self, start, dataclass):
        if not self.option.backward:
            return dataclass(self.option.batch_size, start)
        else:
            a,b,c =  dataclass(self.option.batch_size, start)
            return reverse_seq(a,b,c, self.option.dict_size)

    def next_test(self):
        data, length, target = self._next_batch(self.test_start, self.test_data)
        data  =   torch.tensor(data, dtype=torch.long)
        length  = torch.tensor(length, dtype=torch.long)
        target  = torch.tensor(target, dtype=torch.long)

        self.test_start = (self.test_start+1)% (self.test_data.length/self.option.batch_size)
        return data, length, target

    def next_valid(self):
        data, length, target = self._next_batch(self.valid_start, self.valid_data)
        self.valid_start = (self.valid_start+1)% (self.valid_data.length/self.option.batch_size)
        data  =   torch.tensor(data, dtype=torch.long)
        length  = torch.tensor(length, dtype=torch.long)
        target  = torch.tensor(target, dtype=torch.long)
        return data, length, target

    def next_train(self):
        data, length, target = self._next_batch(self.train_start, self.train_data)
        data  =   torch.tensor(data, dtype=torch.long)
        length  = torch.tensor(length, dtype=torch.long)
        target  = torch.tensor(target, dtype=torch.long)

        self.train_start = (self.train_start+1)% (self.train_data.length/self.option.batch_size)
        return data, length, target
        
def array_data(data,  max_length, dict_size, shuffle=False):
    max_length_m1=max_length-1
    if shuffle==True:
        np.random.shuffle(data)
    sequence_length_pre=np.array([len(line) for line in data]).astype(np.int32)
    sequence_length=[]
    for item in sequence_length_pre:
        if item>max_length_m1:
            sequence_length.append(max_length)
        else:
            sequence_length.append(item+1)
    sequence_length=np.array(sequence_length)
    for i in range(len(data)):
        if len(data[i])>=max_length_m1:
            data[i]=data[i][:max_length_m1]
        else:
            for j in range(max_length_m1-len(data[i])):
                data[i].append(dict_size+1)
        data[i].append(dict_size+1)
    target=np.array(data).astype(np.int32)
    input=np.concatenate([np.ones([len(data), 1])*(dict_size+2), target[:, :-1]], axis=1).astype(np.int32)
    return dataset(input, sequence_length, target)

class dataset(object):
    def __init__(self, input, sequence_length, target ):
        self.input=input
        self.target=target
        self.sequence_length=sequence_length
        self.length=len(input)

    def __call__(self, batch_size, step):
        batch_num=self.length//batch_size+1
        step=step%batch_num
        return self.input[step*batch_size: (step+1)*batch_size], self.sequence_length[step*batch_size: (step+1)*batch_size], self.target[step*batch_size: (step+1)*batch_size]


def reverse_seq(input, sequence_length, target, dict_size):
    batch_size=input.shape[0]
    num_steps=input.shape[1]
    input_new=np.zeros([batch_size, num_steps])+dict_size+1
    target_new=np.zeros([batch_size, num_steps])+dict_size+1
    for i in range(batch_size):
        length=sequence_length[i]-1
        for j in range(length):
            target_new[i][j]=target[i][length-1-j]
        input_new[i][0]=dict_size+2
        for j in range(length):
            input_new[i][j+1]=input[i][length-j]
    return input_new.astype(np.int32), sequence_length.astype(np.int32), target_new.astype(np.int32)
