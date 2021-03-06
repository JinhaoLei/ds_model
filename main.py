import numpy
import tensorflow as tf
import pickle
import time
import random
import preprocess
import util
import os
from score import start
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, BasicLSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
class PTBModel(object):
    def __init__(self, is_training, use_dropout, max_pooling, batch_size):
        with tf.variable_scope('main'):
            self.x = tf.placeholder(name='x', dtype=tf.int32, shape=(batch_size, FLAGS.mslen))
            self.y = tf.placeholder(name='y', dtype=tf.int32, shape=(batch_size, ))
            self.length = tf.placeholder(name='length', dtype=tf.int64, shape=(batch_size))
            pretrained_embedding = pickle.load(open(FLAGS.word_vec_pkl_file))
            pretrained_embedding = numpy.array(pretrained_embedding, dtype='float32')
            print(numpy.shape(pretrained_embedding))
            self.embedding = tf.get_variable(name='embedding',  dtype=tf.float32, initializer=pretrained_embedding, trainable=True)
            inputs = tf.nn.embedding_lookup(self.embedding, self.x)
            print('embedding finished')
            if use_dropout:
                inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)
            flstm_cell = BasicLSTMCell(FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True)
            blstm_cell = BasicLSTMCell(FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True)
                
            if use_dropout:
                flstm_cell = DropoutWrapper(flstm_cell, output_keep_prob=FLAGS.keep_prob)
                blstm_cell = DropoutWrapper(blstm_cell, output_keep_prob=FLAGS.keep_prob)
                
            self._initial_state_f = flstm_cell.zero_state(FLAGS.batch_size, tf.float32)
            self._initial_state_b = blstm_cell.zero_state(FLAGS.batch_size, tf.float32)
                
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    flstm_cell, blstm_cell, inputs, sequence_length=self.length,
                dtype=tf.float32, initial_state_fw =flstm_cell.zero_state(batch_size, tf.float32), initial_state_bw = blstm_cell.zero_state(batch_size, tf.float32))
            outputs = tf.concat(outputs, 2) #batach FLAGS.mslen hidden
            output = tf.reduce_max(outputs, 1)#batch hidden
              
            print('bi-LSTM finished')

               

            w = tf.get_variable('w', [2 * FLAGS.hidden_size, 42], tf.float32)
            b = tf.get_variable('b', [1, 42], tf.float32)
            tmatrix = tf.tanh(tf.matmul(output, w) + b)
                

            pre_dis = tf.nn.softmax(tmatrix)
            #print(pre_dis)
            self.pre_label = tf.argmax(pre_dis, -1)
            #y_mask = tf.one_hot(self.y, 53)
                
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_dis, labels=self.y)
            self.loss = tf.reduce_sum(loss) / batch_size




                
            if not is_training:
                return
            self._lr = tf.Variable(0.0, trainable=False)
      
            tvars = tf.trainable_variables()
    
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
    
            optimizer = tf.train.AdagradOptimizer(self._lr)
        
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
            self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
            print('init finished')

    def assign_lr(self, session, lr_value):

        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def lr(self):       
        return self._lr

def sta(pre_label,gold_label):
    dic = {}
    with open('re_map.csv') as f:
        for n, line in enumerate(f):
            s = line.split('\t')
            dic[s[1].strip()] = s[0]
    f = open('pre.csv', 'w')
    for i in pre_label:
        f.write(dic[str(i)] + '\n')
    f.close()
    f = open('gold.csv', 'w')
    for i in gold_label:
        for j in i:
            f.write(dic[str(j)] + '\n')
    f.close()
    return start()

def run_epoch(session, model, data, label, length, test=False):
    if test == True:
        batch_size = 1
    else:
        batch_size = FLAGS.batch_size
    epoch_size = (len(data) // batch_size)
    statef = session.run(model._initial_state_f)
    stateb = session.run(model._initial_state_b)

    data = data[:len(data)-len(data)%batch_size]
    total = len(data)
    label = label[:len(label)-len(label)%batch_size]
    length = length[:len(length)-len(length)%batch_size]
    total_loss = 0.0
    total_correct = 0
    pre = []
    sids = list(range(len(data)))
    if test == False:
        random.shuffle(sids)
        adata=data
        alabel=label
        lengths=length
        for index in range(len(sids)):
            adata[index]=data[sids[index]]
            alabel[index]=label[sids[index]]
            lengths[index]=length[sids[index]]
        adata=numpy.array(adata)
        adata=numpy.split(adata, epoch_size, 0)
        alabel=numpy.array(alabel)     
        alabel=numpy.split(alabel, epoch_size, 0)
        final_length=numpy.array(lengths)
        final_length=numpy.split(final_length, epoch_size, 0)
    else:
        adata=numpy.array(data)
        adata=numpy.split(adata, epoch_size,0)
        alabel=numpy.array(label)
        alabel=numpy.split(alabel, epoch_size,0)


        final_length=numpy.array(length)
        final_length=numpy.split(final_length, epoch_size,0)


    for ii in range(epoch_size):
        

        feed_dict = {}
        feed_dict[model.x] = adata[ii]
        feed_dict[model.y] = alabel[ii]
        feed_dict[model.length] = final_length[ii]
        if test == False:
            #fetches = [model.loss, model.pre_label]
            fetches = [model.loss, model.pre_label, model._train_op]
            loss, pre_label, _ =session.run(fetches, feed_dict) 
            #loss, pre_label = session.run(fetches, feed_dict)
            total_loss += loss
            pre.extend(pre_label)
            #total_correct += sta(pre_label, alabel[ii])
        else:
            fetches = [model.loss, model.pre_label]
            loss, pre_label =session.run(fetches, feed_dict) 
            total_loss += loss
            pre.extend(pre_label)
            #total_correct += sta(pre_label, alabel[ii])
    #print(numpy.shape(alabel))
    p, r, f1 = sta(pre, alabel)
           
    loss = total_loss / float(epoch_size)
    print("P: %.3f R: %.3f F1: %.3f"%(p, r, f1))
    print("loss:%.7f "%(loss))
    return f1


if __name__=='__main__':
  
    #preprocess.build_vocab()
    #preprocess.convert_to_id()
    #preprocess.prepare_word_vec()
    tf.app.flags.DEFINE_float("init_scale", 0.05, "initialize range")
    tf.app.flags.DEFINE_string("sent_pkl_file", "sen_pkl_file.csv", "sentence file")
    tf.app.flags.DEFINE_string("word_vec_pkl_file", "word_vector.csv", "word embedding file")
    tf.app.flags.DEFINE_float("lr", 1.0, "learning rate")
    tf.app.flags.DEFINE_float("lr_decay", 0.9, "lr decay")
    tf.app.flags.DEFINE_float("keep_prob", 0.5, "keep probability")
    tf.app.flags.DEFINE_integer("batch_size", 50, "batch_size")
    tf.app.flags.DEFINE_integer("max_max_epoch", 30, "iteration")
    tf.app.flags.DEFINE_integer("max_epoch", 20, "iteration")
    tf.app.flags.DEFINE_integer("hidden_size", 50, "hidden size")
    tf.app.flags.DEFINE_float("max_grad_norm", 5.0, "max grad")
    tf.app.flags.DEFINE_boolean("max_pooling", True, "max pooling or mean pooling")
    tf.app.flags.DEFINE_integer("mslen", 96, "uniform sentence length")

    FLAGS = tf.app.flags.FLAGS

    sentences = pickle.load(open(FLAGS.sent_pkl_file))
    #print(sentences[0])
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True 
    train_data_1, train_label_1, train_len_1, valid_data_1, valid_label_1, valid_len_1, test_data, test_label, test_len = sentences
    for cross_label in range(1, 2):
        with tf.Graph().as_default(), tf.Session(config=_config) as session:
            initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
            #saver_path = saver.save(session, "/home/leijh/model1/extract.ckpt")
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, use_dropout=True, max_pooling = FLAGS.max_pooling, batch_size = FLAGS.batch_size)   
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, use_dropout=False, max_pooling = FLAGS.max_pooling, batch_size = 1)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, use_dropout=False, max_pooling = FLAGS.max_pooling, batch_size = 1)
        #saver=tf.train.Saver()
        #ckpt = tf.train.get_checkpoint_state("/home/leijh/model1/exp66")        
        #saver.restore(session,ckpt.model_checkpoint_path)
    #summary_writer = tf.train.SummaryWriter('/home/leijinhao/model1',session.graph)
            tf.initialize_all_variables().run()
        #ckpt = tf.train.get_checkpoint_state("/home/leijh/model1/e")        
        #saver.restore(session,ckpt.model_checkpoint_path)
            if cross_label == 1:
                train_data = train_data_1
                valid_data = valid_data_1
                train_len = train_len_1
                valid_len = valid_len_1
                train_label = train_label_1
                valid_label = valid_label_1
            elif cross_label == 2:
                train_data = train_data_2
                valid_data = valid_data_2
                train_len = train_len_2
                valid_len = valid_len_2
                train_label = train_label_2
                valid_label = valid_label_2
            else:
                train_data = train_data_3
                valid_data = valid_data_3
                train_len = train_len_3
                valid_len = valid_len_3
                train_label = train_label_3
                valid_label = valid_label_3
            maxvalid = -1.0
            #print(len(train_data))
            for i in range(FLAGS.max_max_epoch):   #
                if i>=20:
                    if valid_per<=maxvalid:
                        lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0) 
                        m.assign_lr(session, FLAGS.lr * FLAGS.lr_decay)
                else:
                    m.assign_lr(session, FLAGS.lr)      
            #  llr_decay = 0.5^(i-max_epoch)
                #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                #m.assign_lr(session, config.learning_rate * lr_decay) # learning rate
                #m.assign_lr(session, FLAGS.lr)
                print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(m.lr)))
                print("train-%d:"%(cross_label))
                train_per = run_epoch(session, m, train_data, train_label, train_len)
    
                print("valid-%d:"%(cross_label))
                valid_per = run_epoch(session, mvalid, valid_data, valid_label, valid_len, test=True)
            #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            
                
                #    maxvalid = valid_per
                #    print("test:")
                #    test_result = run_epoch(session, mtest, test_data, test_lable_s, test_lable_e, test_len, test_g, test = True)
