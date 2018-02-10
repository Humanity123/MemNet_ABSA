import os
import pprint
import tensorflow as tf

from nltk import word_tokenize
from data import *
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 300, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "data/glove.840B.300d.txt", "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
flags.DEFINE_string("train_data", "data/Laptops_Train.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("test_data", "data/Laptops_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
  source_count, target_count = [], []
  source_word2idx, target_word2idx, word_set = {}, {}, {}
  max_sent_len = -1
  
  max_sent_len = get_dataset_resources(FLAGS.train_data, source_word2idx, target_word2idx, word_set, max_sent_len)
  max_sent_len = get_dataset_resources(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)
  embeddings = load_embedding_file(FLAGS.pretrain_file, word_set)

  train_data = get_dataset(FLAGS.train_data, source_word2idx, target_word2idx, embeddings)
  test_data = get_dataset(FLAGS.test_data, source_word2idx, target_word2idx, embeddings)

  print "train data size - ", len(train_data[0])
  print "test data size - ", len(test_data[0])

  print "max sentence length - ",max_sent_len
  FLAGS.pad_idx = source_word2idx['<pad>']
  FLAGS.nwords = len(source_word2idx)
  FLAGS.mem_size = max_sent_len

  pp.pprint(flags.FLAGS.__flags)

  print('loading pre-trained word vectors...')
  print('loading pre-trained word vectors for train and test data')
  
  FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,  target_word2idx, FLAGS.edim)
  
  with tf.Session() as sess:
    model = MemN2N(FLAGS, sess)
    model.build_model()
    model.run(train_data, test_data)  

if __name__ == '__main__':
  tf.app.run()  