import os
import sys
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange
import time as tim

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm
        self.pad_idx = config.pad_idx
        self.pre_trained_context_wt = config.pre_trained_context_wt
        self.pre_trained_target_wt = config.pre_trained_target_wt

        self.input = tf.placeholder(tf.int32, [self.batch_size, 1], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.int64, [self.batch_size], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.mem_size], name="mask")
        self.neg_inf = tf.fill([self.batch_size, self.mem_size], -1*np.inf, name="neg_inf")

        self.show = config.show

        self.hid = []

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
      self.global_step = tf.Variable(0, name="global_step")

      self.A = tf.Variable(tf.random_uniform([self.nwords, self.edim], minval=-0.01, maxval=0.01))
      self.ASP = tf.Variable(tf.random_uniform([self.pre_trained_target_wt.shape[0], self.edim], minval=-0.01, maxval=0.01))
      self.C = tf.Variable(tf.random_uniform([self.edim, self.edim], minval=-0.01, maxval=0.01))
      self.C_B =tf.Variable(tf.random_uniform([1, self.edim], minval=-0.01, maxval=0.01))
      self.BL_W = tf.Variable(tf.random_uniform([2 * self.edim, 1], minval=-0.01, maxval=0.01))
      self.BL_B = tf.Variable(tf.random_uniform([1, 1], minval=-0.01, maxval=0.01))

      self.Ain_c = tf.nn.embedding_lookup(self.A, self.context)
      self.Ain = self.Ain_c

      self.ASPin = tf.nn.embedding_lookup(self.ASP, self.input)
      self.ASPout2dim = tf.reshape(self.ASPin, [-1, self.edim])
      self.hid.append(self.ASPout2dim)

      for h in xrange(self.nhop):
        '''
        Bi-linear scoring function for a context word and aspect term
        '''
        self.til_hid = tf.tile(self.hid[-1], [1, self.mem_size])
        self.til_hid3dim = tf.reshape(self.til_hid, [-1, self.mem_size, self.edim])
        self.a_til_concat = tf.concat(axis=2, values=[self.til_hid3dim, self.Ain])
        self.til_bl_wt = tf.tile(self.BL_W, [self.batch_size, 1])
        self.til_bl_3dim = tf.reshape(self.til_bl_wt, [self.batch_size,  2 * self.edim, -1])
        self.att = tf.matmul(self.a_til_concat, self.til_bl_3dim)
        self.til_bl_b = tf.tile(self.BL_B, [self.batch_size, self.mem_size])
        self.til_bl_3dim = tf.reshape(self.til_bl_b, [-1, self.mem_size, 1])
        self.g = tf.nn.tanh(tf.add(self.att, self.til_bl_3dim))
        self.g_2dim = tf.reshape(self.g, [-1, self.mem_size])
        self.masked_g_2dim = tf.add(self.g_2dim, self.mask)
        self.P = tf.nn.softmax(self.masked_g_2dim)
        self.probs3dim = tf.reshape(self.P, [-1, 1, self.mem_size])


        self.Aout = tf.matmul(self.probs3dim, self.Ain)
        self.Aout2dim = tf.reshape(self.Aout, [self.batch_size, self.edim])

        Cout = tf.matmul(self.hid[-1], self.C)
        til_C_B = tf.tile(self.C_B, [self.batch_size, 1])
        Cout_add = tf.add(Cout, til_C_B)
        self.Dout = tf.add(Cout_add, self.Aout2dim)

        if self.lindim == self.edim:
            self.hid.append(self.Dout)
        elif self.lindim == 0:
            self.hid.append(tf.nn.relu(self.Dout))
        else:
            F = tf.slice(self.Dout, [0, 0], [self.batch_size, self.lindim])
            G = tf.slice(self.Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
            K = tf.nn.relu(G)
            self.hid.append(tf.concat(axis=1, values=[F, K]))

    def build_model(self):
      self.build_memory()

      self.W = tf.Variable(tf.random_uniform([self.edim, 3], minval=-0.01, maxval=0.01))
      self.z = tf.matmul(self.hid[-1], self.W)
      
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.z, labels=self.target)

      self.lr = tf.Variable(self.current_lr)
      self.opt = tf.train.AdagradOptimizer(self.lr)

      params = [self.A, self.C, self.C_B, self.W, self.BL_W, self.BL_B]

      self.loss = tf.reduce_sum(self.loss) 

      grads_and_vars = self.opt.compute_gradients(self.loss,params)
      clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                for gv in grads_and_vars]

      inc = self.global_step.assign_add(1)
      with tf.control_dependencies([inc]):
          self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

      tf.initialize_all_variables().run()

      self.correct_prediction = tf.argmax(self.z, 1)

    def train(self, data):
      source_data, source_loc_data, target_data, target_label= data
      N = int(math.ceil(len(source_data) / self.batch_size))
      cost = 0

      x = np.ndarray([self.batch_size, 1], dtype=np.int32)
      time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      target = np.zeros([self.batch_size], dtype=np.int32) 
      context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      mask = np.ndarray([self.batch_size, self.mem_size])
      
      if self.show:
        from utils import ProgressBar
        bar = ProgressBar('Train', max=N)

      rand_idx, cur = np.random.permutation(len(source_data)), 0
      for idx in xrange(N):
        if self.show: bar.next()
        
        context.fill(self.pad_idx)
        time.fill(self.mem_size)
        target.fill(0)
        mask.fill(-1.0*np.inf)
        

        for b in xrange(self.batch_size):
            m = rand_idx[cur]
            x[b][0] = target_data[m]
            target[b] = target_label[m]
            time[b,:len(source_loc_data[m])] = source_loc_data[m]
            context[b,:len(source_data[m])] = source_data[m]
            mask[b,:len(source_data[m])].fill(0)
            cur = cur + 1
 
        z, a, loss, self.step = self.sess.run([ self.z, self.optim,
                                            self.loss,
                                            self.global_step],
                                            feed_dict={
                                                self.input: x,
                                                self.time: time,
                                                self.target: target,
                                                self.context: context,
                                                self.mask: mask})
        
        
       
        if idx%500 == 0:
            print "loss - ", loss

        cost += np.sum(loss)
      
      if self.show: bar.finish()
      _, train_acc = self.test(data)
      return cost/N/self.batch_size, train_acc

    def test(self, data):
      source_data, source_loc_data, target_data, target_label = data
      N = int(math.ceil(len(source_data) / self.batch_size))
      cost = 0

      x = np.ndarray([self.batch_size, 1], dtype=np.int32)
      time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      target = np.zeros([self.batch_size], dtype=np.int32) 
      context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      mask = np.ndarray([self.batch_size, self.mem_size])

      context.fill(self.pad_idx)

      m, acc = 0, 0
      for i in xrange(N):
        target.fill(0)
        time.fill(self.mem_size)
        context.fill(self.pad_idx)
        mask.fill(-1.0*np.inf)
        
        raw_labels = []
        for b in xrange(self.batch_size):
          x[b][0] = target_data[m]
          target[b] = target_label[m]
          time[b,:len(source_loc_data[m])] = source_loc_data[m]
          context[b,:len(source_data[m])] = source_data[m]
          mask[b,:len(source_data[m])].fill(0)
          raw_labels.append(target_label[m])
          m += 1

        loss = self.sess.run([self.loss],
                                        feed_dict={
                                            self.input: x,
                                            self.time: time,
                                            self.target: target,
                                            self.context: context,
                                            self.mask: mask})
        cost += np.sum(loss)

        predictions = self.sess.run(self.correct_prediction, feed_dict={self.input: x,
                                                     self.time: time,
                                                     self.target: target,
                                                     self.context: context,
                                                     self.mask: mask})

        for b in xrange(self.batch_size):
          if raw_labels[b] == predictions[b]:
            acc = acc + 1

      return cost, acc/float(len(source_data))

    def run(self, train_data, test_data):
      print('training...')
      self.sess.run(self.A.assign(self.pre_trained_context_wt))
      self.sess.run(self.ASP.assign(self.pre_trained_target_wt))

      for idx in xrange(self.nepoch):
        print('epoch '+str(idx)+'...')
        train_loss, train_acc = self.train(train_data)
        test_loss, test_acc = self.test(test_data)
        print('train-loss=%.2f;train-acc=%.2f;test-acc=%.2f;' % (train_loss, train_acc, test_acc))
        self.log_loss.append([train_loss, test_loss])
        