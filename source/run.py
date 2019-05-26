import os
import numpy as np
import time,random
from sampling import *
import argparse
from utils import Option

def main():

    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="3", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)

    # data property
    parser.add_argument('--data_path', default='data/quoradata/test.txt', type=str)
    parser.add_argument('--dict_path', default='data/quoradata/dict.pkl', type=str)
    parser.add_argument('--dict_size', default=30000, type=int)
    parser.add_argument('--vocab_size', default=30003, type=int)
    parser.add_argument('--backward', default=False, action="store_true")
    parser.add_argument('--keyword_pos', default=True, action="store_false")
    # model architecture
    parser.add_argument('--num_steps', default=15, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--emb_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=300, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--model', default=0, type=int)
    # optimization
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--clip_norm', default=0.00, type=float)
    parser.add_argument('--no_cuda', default=False, action="store_true")
    parser.add_argument('--local', default=False, action="store_true")
    parser.add_argument('--threshold', default=0.1, type=float)

    # evaluation
    parser.add_argument('--sim', default='word_max', type=str)
    parser.add_argument('--mode', default='sa', type=str)
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--accumulate_step', default=1, type=int)
    parser.add_argument('--backward_path', default=None, type=str)
    parser.add_argument('--forward_path', default=None, type=str)

    # sampling
    parser.add_argument('--use_data_path', default='data/quoradata/test.txt', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--pos_path', default='POS/english-models', type=str)
    parser.add_argument('--emb_path', default='data/quoradata/emb.pkl', type=str)
    parser.add_argument('--max_key', default=3, type=float)
    parser.add_argument('--max_key_rate', default=0.5, type=float)
    parser.add_argument('--rare_since', default=30000, type=int)
    parser.add_argument('--sample_time', default=100, type=int)
    parser.add_argument('--search_size', default=100, type=int)
    parser.add_argument('--action_prob', default=[0.3,0.3,0.3,0.3], type=list)
    parser.add_argument('--just_acc_rate', default=0.0, type=float)
    parser.add_argument('--sim_mode', default='keyword', type=str)
    parser.add_argument('--save_path', default='temp.txt', type=str)
    parser.add_argument('--forward_save_path', default='data/tfmodel/forward.ckpt', type=str)
    parser.add_argument('--backward_save_path', default='data/tfmodel/backward.ckpt', type=str)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--keep_prob', default=1, type=float)
    parser.add_argument('--N_repeat', default=1, type=int)
    parser.add_argument('--C', default=0.03, type=float)
    parser.add_argument('--M_kw', default=8, type=float)
    parser.add_argument('--M_bleu', default=1, type=float)

    d = vars(parser.parse_args())
    option = Option(d)

    random.seed(option.seed)
    np.random.seed(option.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    config = option

    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)


    if option.batch_size==1:
        simulatedAnnealing(option)
    else:
        simulatedAnnealing_batch(option)

    print("="*36 + "Finish" + "="*36)

if __name__ == "__main__":
    main()

