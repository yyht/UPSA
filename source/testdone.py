from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time, random
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import argparse
from tensorflow.python.client import device_lib
import os 
from utils import *


class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))



logging = tf.logging

def data_type():
  return  tf.float32

class PTBModel(object):
  #The language model.

  def __init__(self, is_training, is_test_LM=False):
    self._is_training = is_training
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    self._input=tf.placeholder(shape=[None, config.num_steps], dtype=tf.int32)
    self._target=tf.placeholder(shape=[None, config.num_steps], dtype=tf.int32)
    self._sequence_length=tf.placeholder(shape=[None], dtype=tf.int32)
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input)
    softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    output = self._build_rnn_graph(inputs, self._sequence_length, is_training)

    output=tf.reshape(output, [-1, config.hidden_size])
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
                                      config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer()
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

  def _build_rnn_graph(self, inputs, sequence_length, is_training):
    return self._build_rnn_graph_lstm(inputs, sequence_length, is_training)

  def _get_lstm_cell(self, is_training):
    return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)

  def _build_rnn_graph_lstm(self, inputs, sequence_length, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell( is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
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

def main(config):

  if config.mode=='forward' or config.mode=='use':
    with tf.name_scope("forward_train"):
      with tf.variable_scope("forward", reuse=None):
        m_forward = PTBModel(is_training=True)
    with tf.name_scope("forward_test"):
      with tf.variable_scope("forward", reuse=True):
        mtest_forward = PTBModel(is_training=False)
    var=tf.trainable_variables()
    var_forward=[x for x in var if x.name.startswith('forward')]
    saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
  if config.mode=='backward' or config.mode=='use':
    with tf.name_scope("backward_train"):
      with tf.variable_scope("backward", reuse=None):
        m_backward = PTBModel(is_training=True)
    with tf.name_scope("backward_test"):
      with tf.variable_scope("backward", reuse=True):
        mtest_backward = PTBModel(is_training=False)
    var=tf.trainable_variables()
    var_backward=[x for x in var if x.name.startswith('backward')]
    saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
  init = tf.global_variables_initializer()
  

  with tf.Session() as session:
    session.run(init)
    input = [[3,4,5,6,6,7,8,9,4,5,6,7,8,9,2]]
    sequence_length = [10]
    prob_old=run_epoch(session, mtest_forward, input, sequence_length, mode='use')
    print(prob_old)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="3", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)

    # data property
    parser.add_argument('--data_path', default='data/quora/quora.txt', type=str)
    parser.add_argument('--dict_path', default='data/quora/dict.pkl', type=str)
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
    parser.add_argument('--batch_size', default=128, type=int)
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
    parser.add_argument('--use_data_path', default='data/input/input.txt', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--pos_path', default='POS/english-models', type=str)
    parser.add_argument('--emb_path', default='data/quora/emb.pkl', type=str)
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

    d = vars(parser.parse_args())
    option = Option(d)

    random.seed(option.seed)
    np.random.seed(option.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    config = option
    main(option)

