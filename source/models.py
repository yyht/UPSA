import torch
import torch.nn as nn
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

def data_type():
  return  tf.float32


class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, option):
		super(RNNModel, self).__init__()
		rnn_type = 'LSTM'
		self.option = option
		dropout = option.dropout
		ntoken = option.vocab_size
		ninp = option.emb_size
		nhid = option.hidden_size
		self.nlayers = option.num_layers
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.rnn = nn.LSTM(ninp, nhid, self.nlayers, dropout = dropout ,batch_first=True)
		self.decoder = nn.Linear(nhid, ntoken)
		self.init_weights()
		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.criterion = nn.CrossEntropyLoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, target):
		'''
		bs,15; bs,15
		'''
		batch_size = input.size(0)
		length = input.size(1)
		target = target.view(-1)

		emb = self.drop(self.encoder(input))
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		output = self.drop(output).contiguous().view(batch_size*length,-1)
		decoded = self.decoder(output)
		loss = self.criterion(decoded, target)
		v,idx = torch.max(decoded,1)
		acc = torch.mean(torch.eq(idx,target).float())
		return loss,acc, decoded.view(batch_size, length, self.ntoken)

	def predict(self, input):
		'''
		bs,15; bs,15
		'''
		batch_size = input.size(0)
		length = input.size(1)

		emb = self.drop(self.encoder(input))
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		output = self.drop(output).contiguous().view(batch_size*length,-1)
		decoded = nn.Softmax(1)(self.decoder(output))
		return decoded.view(batch_size, length, self.ntoken)



	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class PTBModel(object):
  #The language model.

  def __init__(self, is_training, config, is_test_LM=False):
    self._is_training = is_training
    self.config = config
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
          self.config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)

  def _build_rnn_graph_lstm(self, inputs, sequence_length, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell( is_training)
      if is_training and self.config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=self.config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
    outputs, states=tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=data_type())

    return outputs
 
