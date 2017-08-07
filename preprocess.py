#encoding=utf-8

import os
import random
import re
import collections

import pickle
import numpy
import fileinput

import config

import util




#@util.file_checker(config.vocab_file)
def build_vocab():
    counter = collections.Counter()
    f = fileinput.input(files=dataset_file)
    linenumber=0
    maxlen = 0
    for line in f:
        words = line.split('\t')[5].split()
        counter.update(words)
        if len(words) > maxlen:
            maxlen = len(words)
        linenumber+=1    
    f.close()
    print(maxlen)
    with open(vocab_file, 'w') as vocab_file:
        for w, n in counter.most_common():
            vocab_file.write('%s %d\n' % (w, n))

def relation_2id():
    relation_map = {}
    with open(relation2id_file) as f:
        for n, line in enumerate(f):
            s = line.split()
            relation_map[s[0]] = s[1].strip()


#@util.file_checker(config.lm_sent_pkl_file)
def convert_to_id():
    word_to_id = util.load_word_to_id()
    unk_id = 2286
    sents = []
    labels=[]
    lengths=[]
    with open(train_data_file) as lm_sent_txt_file:
        for n, line in enumerate(lm_sent_txt_file):
            ids = [word_to_id.get(w, unk_id) for w in line.split('\t')[5].split()]
            sents.append(ids)
            label = relation_map[line.split('\t')[4]]
            labels.append(label)
            length = len(ids)
            lengths.append(length)
    
    sids = list(range(len(sents)))
    random.shuffle(sids)
    three = len(sents) / 3
    train_data_1 = [sents[si] for si in sids[:2 * three]]
    train_label_1=[labels[si] for si in sids[:2 * three]]
    train_len_1=[lengths[si] for si in sids[:2 * three]]
    train_data_2 = [sents[si] for si in sids[three:]]
    train_label_2=[labels[si] for si in sids[three:]]
    train_len_2=[lengths[si] for si in sids[three:]]
    first_three = sids[:three]
    first_three.extend(sids[2 * three:])
    train_data_3 = [sents[si] for si in first_three]
    train_label_3=[labels[si] for si in first_three]
    train_len_3=[lengths[si] for si in first_three]

    valid_data_1 = [sents[si] for si in sids[2 * three:]]
    valid_label_1=[labels[si] for si in sids[2 * three:]]
    valid_len_1=[lengths[si] for si in sids[2 * three:]]

    valid_data_2 = [sents[si] for si in sids[:three]]
    valid_label_2=[labels[si] for si in sids[:three]]
    valid_len_2=[lengths[si] for si in sids[:three]]
    
    
    valid_data_3 = [sents[si] for si in sids[three:2 * three]]
    valid_label_3=[labels[si] for si in sids[three:2 * three]]
    valid_len_3=[lengths[si] for si in sids[three:2 * three]]

    sents = []
    labels=[]
    lengths=[]

    with open(test_data_file) as lm_sent_txt_file:
        for n, line in enumerate(lm_sent_txt_file):
            ids = [word_to_id.get(w, unk_id) for w in line.split('\t')[5].split()]
            sents.append(ids)
            label = relation_map[line.split('\t')[4]]
            labels.append(label)
            length = len(ids)
            lengths.append(length)
    
    sids = list(range(len(sents)))
    test_data= [sents[si] for si in sids]
    test_label=[labels[si] for si in sids]
    test_len=[lengths[si] for si in sids]
    
    #for i in range(10):
        #print(test[i],testlen[i],testgold[i],testcategory[i],testgold[i])
    with open(lm_sent_pkl_file, 'wb') as lm_sent_pkl_file:
        pickle.dump((train_data_1, train_data_2, train_data_3, train_label_1, train_label_2, train_label_3, 
    train_len_1, train_len_2, train_len_3, valid_data_1, valid_data_2, valid_data_3, 
    valid_label_1, valid_label_2, valid_label_3, valid_len_1, valid_len_2, valid_len_3,
    test_data, test_label, test_len, testle,trainlen, validlen, testlen,traingold,validgold, testgold), lm_sent_pkl_file, pickle.HIGHEST_PROTOCOL)

#@util.file_checker(config.word_vec_pkl_file)
def prepare_word_vec():
    word_to_id = util.load_word_to_id()
    vec = numpy.random.randn(len(word_to_id), word_vec_size)
    with open(word_vec_txt_file) as vec_file:
        for line in vec_file:
            its = line.split()
            if len(its) != word_vec_size + 1:
                continue
            w = its[0]
            v = [float(t) for t in its[1:]]
            if w in word_to_id:
                vec[word_to_id[w]] = v
    #print(word_to_id)
    with open(word_vec_pkl_file, 'wb') as pkl_file:
        pickle.dump(vec, pkl_file, pickle.HIGHEST_PROTOCOL)

def main():
  #  prepare_sentence()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--word_size', dest='word_vec_size', default=50)
    parser.add_argument('--test_file', dest='test_data_file', default='test.txt')
    parser.add_argument('--train_file', dest='train_data_file', default='train.txt')
    parser.add_argument('--word_vec_out_file', dest='word_vec_pkl_file', default='word_vector.csv')
    parser.add_argument('--word_vec_txt', dest='word_vec_txt_file', default='nyt50.txt')
    parser.add_argument('--relation2id_file', dest='relation2id_file', default='relation2id.csv')
    parser.add_argument('--vocab_file', dest='vocab_file', default='vocabulary.csv')
    parser.add_argument('--all_file', dest='dataset_file', default='all.csv')
    parser.add_argument('--sen_pkl_file', dest='lm_sent_pkl_file', default='sen_pkl_file.csv')
    args = parser.parse_args()
    relation_2id()
    build_vocab()
    convert_to_id()
    prepare_word_vec()


if __name__ == '__main__':
    main()
