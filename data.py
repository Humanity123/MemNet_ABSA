from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

import numpy as np
import os
import xml.etree.ElementTree as ET
import html
import HTMLParser
import re

stop = set(stopwords.words('english')) 

def load_embedding_file(embed_file_name, word_set):
  ''' loads embedding file and returns a dictionary (word -> embedding) for the words existing in the word_set '''

  embeddings = {}
  with open(embed_file_name, 'r') as embed_file:
    for line in embed_file:
      content = line.strip().split()
      word = content[0]
      if word in word_set:
        embedding = np.array(content[1:], dtype=float)
        embeddings[word] = embedding

  return embeddings

def get_dataset_resources(data_file_name, sent_word2idx, target_word2idx, word_set, max_sent_len):
  ''' updates word2idx and word_set '''
  if len(sent_word2idx) == 0:
    sent_word2idx["<pad>"] = 0

  word_count = []
  sent_word_count = []
  target_count = []

  words = []
  sentence_words = []
  target_words = []

  with open(data_file_name, 'r') as data_file:
    lines = data_file.read().split('\n')
    for line_no in range(0, len(lines)-1, 3):
      sentence = lines[line_no]
      target = lines[line_no+1]

      sentence.replace("$T$", "")
      sentence = sentence.lower()
      target = target.lower()
      max_sent_len = max(max_sent_len, len(sentence.split()))
      sentence_words.extend(sentence.split())
      target_words.extend([target])
      words.extend(sentence.split() + target.split())

    sent_word_count.extend(Counter(sentence_words).most_common())
    target_count.extend(Counter(target_words).most_common())
    word_count.extend(Counter(words).most_common())

    for word, _ in sent_word_count:
      if word not in sent_word2idx:
        sent_word2idx[word] = len(sent_word2idx)

    for target, _ in target_count:
      if target not in target_word2idx:
        target_word2idx[target] = len(target_word2idx)    

    for word, _ in word_count:
      if word not in word_set:
        word_set[word] = 1

  return max_sent_len

def get_embedding_matrix(embeddings, sent_word2idx,  target_word2idx, edim):
  ''' returns the word and target embedding matrix ''' 
  word_embed_matrix = np.zeros([len(sent_word2idx), edim], dtype = float)
  target_embed_matrix = np.zeros([len(target_word2idx), edim], dtype = float)

  for word in sent_word2idx:
    if word in embeddings:
      word_embed_matrix[sent_word2idx[word]] = embeddings[word]

  for target in target_word2idx:
    for word in target:
      if word in embeddings:
        target_embed_matrix[target_word2idx[target]] += embeddings[word]
    target_embed_matrix[target_word2idx[target]] /= max(1, len(target.split()))

  print type(word_embed_matrix)
  return word_embed_matrix, target_embed_matrix


def get_dataset(data_file_name, sent_word2idx, target_word2idx, embeddings):
  ''' returns the dataset'''
  sentence_list = []
  location_list = []
  target_list = []
  polarity_list = []


  with open(data_file_name, 'r') as data_file:
    lines = data_file.read().split('\n')
    for line_no in range(0, len(lines)-1, 3):
      sentence = lines[line_no].lower()
      target = lines[line_no+1].lower()
      polarity = int(lines[line_no+2])

      sent_words = sentence.split()
      target_words = target.split()
      try:
        target_location = sent_words.index("$t$")
      except:
        print "sentence does not contain target element tag"
        exit()

      is_included_flag = 1
      id_tokenised_sentence = []
      location_tokenised_sentence = []
      
      for index, word in enumerate(sent_words):
        if word == "$t$":
          continue
        try:
          word_index = sent_word2idx[word]
        except:
          print "id not found for word in the sentence"
          exit()

        location_info = abs(index - target_location)

        if word in embeddings:
          id_tokenised_sentence.append(word_index)
          location_tokenised_sentence.append(location_info)

        # if word not in embeddings:
        #   is_included_flag = 0
        #   break

      is_included_flag = 0
      for word in target_words:
        if word in embeddings:
          is_included_flag = 1
          break
          

      try:
        target_index = target_word2idx[target]
      except:
        print target
        print "id not found for target"
        exit()


      if not is_included_flag:
        print sentence
        continue

      sentence_list.append(id_tokenised_sentence)
      location_list.append(location_tokenised_sentence)
      target_list.append(target_index)
      polarity_list.append(polarity)

  return sentence_list, location_list, target_list, polarity_list
