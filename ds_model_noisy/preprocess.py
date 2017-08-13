#encoding=utf-8

import os
import random
import re
import collections

import pickle
import numpy
import fileinput
import argparse
#import config

#import util



def load_word(args):
    words = [line.split()[0] for line in open(args.vocab_file)]
    return words


def load_word_to_id(args):
    words = load_word(args)
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

#@util.file_checker(config.vocab_file)
def build_vocab(args):
    counter = collections.Counter()
    f = fileinput.input(files=args.dataset_file)
    linenumber=0
    maxlen = 0
    max_index = -1
    for n, line in enumerate(f): 
        if n % 10000 ==0:
            print("read vocab %d"%(n))
        words = line.strip().split()
        counter.update(words)
        if len(words) >maxlen:
            maxlen = len(words)       
        linenumber+=1    
    f.close()
    print(maxlen)
    #print()
    with open(args.vocab_file, 'w') as vocab_file:
        for w, n in counter.most_common():
            if n>=2:
                vocab_file.write('%s %d\n' % (w, n))

def relation_2id(args):
    relation_map = {}
    with open(args.relation2id_file) as f:
        for n, line in enumerate(f):
            s = line.split()
            #print(s[0])
            relation_map[s[0]] = int(s[1].strip())
    return relation_map

#@util.file_checker(config.lm_sent_pkl_file)
def convert_to_id(args, relation_map):
    word_to_id = load_word_to_id(args)
    unk_id = 2286
    def convert(type, data_file, gold_file):
        sents = []
        labels=[]
        lengths=[]
        no_rel = []
        with open(gold_file) as gold_file:
            for n, line in enumerate(gold_file):
                if n % 10000 ==0:
                    print ('read_%s_gold%d'%(type, n))
                label = relation_map[line.strip()]
                labels.append(label)
        if type == 'train' and args.noisy_per!=0:
            population = []
            #relation_pop = range(42)
            st = 0
            for i in range(len(labels)):
                if labels[i] != 22:
                    population.append(i)
                    st +=1
            #print(st)
            chosen = random.sample(population, int(len(population) * args.noisy_per))
            noisy_labels = labels[:]
            for i in chosen:
                relation_pop = range(42)
                relation_pop.remove(int(labels[i]))
                relation_pop.remove(22)
                noisy_labels[i] = str(random.sample(relation_pop, 1)[0])
                #print(noisy_labels[i])
            print('test noisy')
        else:
            noisy_labels = labels[:]
        def test():
            n = 0
            #print(labels[:10])
            #print(noisy_labels[:10])
            for i in range(len(labels)):
                if labels[i] != noisy_labels[i]:
                    n+=1
            print(n)
            print(float(n/len(labels)))
        #if type == 'train':
        test()
        
        with open(data_file) as lm_sent_txt_file:
            for n, line in enumerate(lm_sent_txt_file):
                if n % 10000 ==0:
                    print ('read_%s_data%d'%(type, n))
                ids = [word_to_id.get(w, unk_id) for w in line.strip().split()]
                length = len(ids)
                ids.extend([unk_id] * (args.max_len - len(ids)))
                sents.append(ids)
                #length = len(ids)
                lengths.append(length)
        return sents, labels, lengths, noisy_labels
     
    
    '''sids = list(range(len(sents)))
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
    valid_len_3=[lengths[si] for si in sids[three:2 * three]]'''


    train_data, train_label, train_len, train_noisy_labels = convert('train', args.train_data_file, args.train_gold)
    valid_data, valid_label, valid_len, valid_noisy_labels = convert('dev', args.dev_data_file, args.dev_gold)
    test_data, test_label, test_len, test_noisy_labels = convert('test', args.test_data_file, args.test_gold)

    print(len(train_data), len(train_label), len(train_len), len(train_noisy_labels))
    #for i in range(3):
    #    print(test[i],testlen[i],testgold[i],testcategory[i],testgold[i])
    with open(args.lm_sent_pkl_file+'%s'%(str(args.noisy_per)), 'wb') as lm_sent_pkl_file:
        pickle.dump((train_data, train_label, train_len, train_noisy_labels, valid_data, valid_label, valid_len, valid_noisy_labels, test_data, test_label, test_len, test_noisy_labels), lm_sent_pkl_file, pickle.HIGHEST_PROTOCOL)

#@util.file_checker(config.word_vec_pkl_file)
def prepare_word_vec(args):
    word_to_id = load_word_to_id(args)
    vec = numpy.random.randn(len(word_to_id), args.word_vec_size)
    with open(args.word_vec_txt_file) as vec_file:
        for line in vec_file:
            its = line.split()
            if len(its) != args.word_vec_size + 1:
                continue
            w = its[0]
            v = [float(t) for t in its[1:]]
            if w in word_to_id:
                vec[word_to_id[w]] = v
    #print(word_to_id)
    with open(args.word_vec_pkl_file, 'wb') as pkl_file:
        pickle.dump(vec, pkl_file, pickle.HIGHEST_PROTOCOL)

def main():
  #  prepare_sentence()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--word_size', dest='word_vec_size', default=200)
    parser.add_argument('--test_file', dest='test_data_file', default='test_data.csv')
    parser.add_argument('--train_file', dest='train_data_file', default='train_data.csv')
    parser.add_argument('--dev_file', dest='dev_data_file', default='dev_data.csv')
    parser.add_argument('--test_gold', dest='test_gold', default='test.gold')
    parser.add_argument('--train_gold', dest='train_gold', default='train.gold')
    parser.add_argument('--dev_gold', dest='dev_gold', default='dev.gold')
    parser.add_argument('--word_vec_out_file', dest='word_vec_pkl_file', default='word_vector.csv')
    parser.add_argument('--word_vec_txt', dest='word_vec_txt_file', default='/scr/leijh/data/glove.6B.200d.txt')
    parser.add_argument('--relation_to_id_file', dest='relation2id_file', default="re_map.csv")
    parser.add_argument('--vocab_file', dest='vocab_file', default='vocabulary.csv')
    parser.add_argument('--all_file', dest='dataset_file', default='all.csv')
    parser.add_argument('--sen_pkl_file', dest='lm_sent_pkl_file', default='sen_pkl_file.csv')
    parser.add_argument('--max_len', dest='max_len', default=95)
    parser.add_argument('--random_noise', dest='noisy_per', default=0.8)
    args = parser.parse_args()
    #relation_map = {}
    relation_map = relation_2id(args)
    #build_vocab(args)
    convert_to_id(args, relation_map)
    #prepare_word_vec(args)


if __name__ == '__main__':
    main()
