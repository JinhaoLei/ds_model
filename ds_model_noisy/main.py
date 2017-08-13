import numpy
import tensorflow as tf
import pickle
import time
import random
#import preprocess
import util
import os
from score import start
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, BasicLSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
class PTBModel(object):
    def __init__(self, is_training, use_dropout, max_pooling, batch_size):
        with tf.variable_scope('main'):
            self.x = tf.placeholder(name='x', dtype=tf.int32, shape=(batch_size, FLAGS.mslen))
            self.y = tf.placeholder(name='y', dtype=tf.int64, shape=(batch_size, ))
            self.length = tf.placeholder(name='length', dtype=tf.int64, shape=(batch_size))
            
            self.weight = tf.placeholder(name='weight', dtype=tf.float32, shape=(1,42))
            self.threshold = tf.placeholder(name='threshold', dtype=tf.float32, shape=(batch_size,))
            pretrained_embedding = pickle.load(open(FLAGS.word_vec_pkl_file))
            pretrained_embedding = numpy.array(pretrained_embedding, dtype='float32')
            print(numpy.shape(pretrained_embedding))
            self.embedding = tf.get_variable(name='embedding',  dtype=tf.float32, initializer=pretrained_embedding, trainable=True)
            inputs = tf.nn.embedding_lookup(self.embedding, self.x)
            print('embedding finished')
            if use_dropout:
                inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)
            flstm_cell = BasicLSTMCell(FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True)
            #blstm_cell = BasicLSTMCell(FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True)
            self._initial_state_f = flstm_cell.zero_state(FLAGS.batch_size, tf.float32)
            if use_dropout:
                flstm_cell = DropoutWrapper(flstm_cell, output_keep_prob=FLAGS.keep_prob)
                #blstm_cell = DropoutWrapper(blstm_cell, output_keep_prob=FLAGS.keep_prob)
            cell = MultiRNNCell([flstm_cell] * 2, state_is_tuple=True)
                
            
            #self._initial_state_b = blstm_cell.zero_state(FLAGS.batch_size, tf.float32)
                
            _, final_states = tf.nn.dynamic_rnn(
                    flstm_cell, inputs, sequence_length=self.length,
                dtype=tf.float32, initial_state =flstm_cell.zero_state(batch_size, tf.float32))
            final_states = tf.concat(final_states, 1) #batach FLAGS.mslen hidden
            #output = tf.reduce_max(outputs, 1)#batch hidden
            #output = tf.Print(output,[output])  
            print('bi-LSTM finished')

            

            w = tf.get_variable('w', [2 * FLAGS.hidden_size, 42], tf.float32)
            b = tf.get_variable('b', [1, 42], tf.float32)
            tmatrix = tf.matmul(final_states, w) + b
            self.tmatrix = tf.matmul(final_states, w) + b
            pre_dis = tf.nn.softmax(tmatrix)
            self.get_confidence = tf.reduce_max(pre_dis,1)
            self.pre_dis = pre_dis
            self.pre_label = tf.argmax(pre_dis, -1)

            if not is_training:
                return
            
            diff = tf.reduce_max(pre_dis, -1) - self.threshold
            flag = tf.sign(tf.sign(tf.sign(diff) - 0.5) + 1)
            to_switch = tf.concat([self.y, self.pre_label], 0)
            rang = tf.range(batch_size)    
            index = rang + batch_size * tf.cast(flag, tf.int32)
            self.new_y = tf.gather(to_switch, index)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tmatrix, labels=self.new_y)
            #loss = loss * weights
            self.loss = tf.reduce_sum(loss) / batch_size




            self._lr = tf.Variable(0.0, trainable=False)
      
            tvars = tf.trainable_variables()
    
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
            print(tf.gradients(self.loss, tvars))
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
    #f = open('pre.csv', 'w')
    no_relation = 0
    pre = []
    #print(pre_label)
    for i in pre_label:
        if i==22:
            no_relation+=1
        pre.append(dic[str(i)])
        #f.write(dic[str(i)] + '\n')
    #f.close()
    #f = open('gold.csv', 'w')
    gold = []
    for i in gold_label:
        for j in i:
            gold.append(dic[str(j)])
    #        f.write(dic[str(j)] + '\n')
    #f.close()
    print(no_relation)
    return start(gold, pre)

def strategy(noisy, clean, new, pre):
    
    correct_change = 0 
    noisy_but_wrong_change = 0
    nonoisy_but_wrong_change = 0
    changen = 0
    #for i in range()
    #print(noisy[:10])
    #print(new[:10])
    for i in range(len(noisy)):
        if new[i] != int(noisy[i]):
            changen += 1
            if noisy[i] != clean[i] and new[i] == clean[i]:
                correct_change += 1
            if noisy[i] != clean[i] and new[i] != clean[i]:
                noisy_but_wrong_change +=1
            if noisy[i] == clean[i]:
                nonoisy_but_wrong_change+=1
    #num_of_noisy = int(num_of_data * FLAGS.noisy_per)
 
    return changen, correct_change, noisy_but_wrong_change, nonoisy_but_wrong_change

def get_training_info():
    info = "Training params:\n"
    info += "\tinit_lr: %g\n" % FLAGS.lr
    info += "\tnum_epoch: %d\n" % FLAGS.max_max_epoch
    info += "\tdecay_epoch: %d\n" % FLAGS.max_epoch
    info += "\tbatch_size: %d\n" % FLAGS.batch_size
    info += "\tnoisy_per: %.1f\n" % FLAGS.noisy_per
    info += "\tthreshold_start_epoch: %d\n" % FLAGS.startepoch

    return info
           
            

def run_epoch(session, model, data, noisy_label, gold_label, length, epoch, test=False):
    if test == True:
        batch_size = 160
    else:
        batch_size = FLAGS.batch_size
    epoch_size = (len(data) // batch_size)
    statef = session.run(model._initial_state_f)
    #stateb = session.run(model._initial_state_b)

    data = data[:len(data)-len(data)%batch_size]
    total = len(data)
    noisy_label = noisy_label[:len(noisy_label)-len(noisy_label)%batch_size]
    gold_label = gold_label[:len(gold_label)-len(gold_label)%batch_size]
    #print(noisy_label)
    length = length[:len(length)-len(length)%batch_size]
    weight = [0 for i in range(42)]
    #for lab in label:
    #    weight[int(lab)]+=1
    #for i in range(len(weight)):
    #    weight[i] = 1/float(weight[i]) * 1500
    #weight = [weight]
    #print(weight)
    #return
    total_loss = 0.0
    confidence = 0.0
    total_correct = 0
    pre = []
    sids = list(range(len(data)))
    if test == False:
        random.shuffle(sids)
        adata=data[:]
        nlabel=noisy_label[:]
        glabel=gold_label[:]

        lengths=length[:]
        for index in range(len(sids)):
            adata[index]=data[sids[index]]
            nlabel[index]=noisy_label[sids[index]]
            glabel[index]=gold_label[sids[index]]
            lengths[index]=length[sids[index]]
        adata=numpy.array(adata)
        adata=numpy.split(adata, epoch_size, 0)
        nlabel=numpy.array(nlabel)     
        nlabel=numpy.split(nlabel, epoch_size, 0)
        glabel=numpy.array(glabel)     
        glabel=numpy.split(glabel, epoch_size, 0)
        final_length=numpy.array(lengths)
        final_length=numpy.split(final_length, epoch_size, 0)
        testn = 0
        testtotal = 0
        for i in range(len(nlabel)):
            for j in range(len(nlabel[i])):
                testtotal +=1
                if int(nlabel[i][j]) != int(glabel[i][j]):
                    testn+=1
        print(testn, testtotal)
        #@print(float(testn/float(testtotal)))
    else:
        adata=numpy.array(data)
        adata=numpy.split(adata, epoch_size,0)
        nlabel=numpy.array(noisy_label)
        nlabel=numpy.split(nlabel, epoch_size,0)


        final_length=numpy.array(length)
        final_length=numpy.split(final_length, epoch_size,0)
    '''weight = []
    for i in range(42):
        if i==22:
            weight.append(1.0)
        else:
            weight.append(1.0)    
    weight = [weight]'''
    correct_changes = 0
    noisy_but_wrong_changes = 0
    nonoisy_but_wrong_changes = 0
    changens = 0
    t = (1.0 - max(0, epoch - FLAGS.startepoch)/100.0) ** FLAGS.lam
    print('threshold:%.3f'%(t))
    for ii in range(epoch_size):
        #print(adata[ii][0])
        #print(alabel[ii][0])
        #print(final_length[ii])
        #break
        #print(ii)
        feed_dict = {}
        feed_dict[model.x] = adata[ii]
        feed_dict[model.y] = nlabel[ii]
        feed_dict[model.length] = final_length[ii]
        if test == False:
            #fetches = [model.loss, model.pre_label]
            #fetches = [model.get_confidence, model.pre_label]
            #confidence, pre_label =session.run(fetches, feed_dict) 
            #t = (1.0 - max(0, epoch - FLAGS.startepoch)/100.0) ** FLAGS.lam
            threshold = []
            for i in range(batch_size):
                threshold.append(t)
            feed_dict[model.threshold] = threshold
            
            #feed_dict[model.y] = new_y
            

            fetches = [model.loss, model.pre_label, model.get_confidence, model.new_y, model._train_op]
            loss, pre_label, conf, new_label, _ = session.run(fetches, feed_dict)
            for i in conf:
                confidence+=i
            changen, correct_change, noisy_but_wrong_change, nonoisy_but_wrong_change = strategy(nlabel[ii], glabel[ii], new_label, pre_label)
            correct_changes += correct_change
            noisy_but_wrong_changes += noisy_but_wrong_change 
            nonoisy_but_wrong_changes += nonoisy_but_wrong_change
            changens += changen
            #for pre_i in range(len(pre_dis)):
            #    print(pre_dis[pre_i][:])
            #    print(pre_dis[pre_i][22])
            total_loss += loss
            #print(pre_label)
            pre.extend(pre_label)
            #total_correct += sta(pre_label, alabel[ii])
            #print("average confidence %.3f / %d %.5f"%(confidence, len(data), confidence/float(len(data))))
        else:
            fetches = [model.pre_label]
            pre_label =session.run(fetches, feed_dict) 
            #total_loss += loss
            pre.extend(pre_label[0])
            #total_correct += sta(pre_label, alabel[ii])
    #print(numpy.shape(alabel))
    if test ==False:
        p, r, f1 = sta(pre, glabel)
    else:
        p, r, f1 = sta(pre, nlabel)       
    loss = total_loss / float(epoch_size)
    print("P: %.3f R: %.3f F1: %.3f"%(p, r, f1))
    print("loss:%.7f "%(loss))
    if test ==False:
        print("average confidence %.3f / %d %.5f"%(confidence, len(data), confidence/float(len(data))))
    if test == False:
        try:
            print("correct_change: %d %.3f noisy_but_wrong_change: %d %.3f nonoisy_but_wrong_change: %d"%(correct_changes, correct_changes / float(changens), noisy_but_wrong_changes, noisy_but_wrong_changes / float(changens), nonoisy_but_wrong_changes))
        except:
            print("correct_change: %d %.3f noisy_but_wrong_change: %d %.3f nonoisy_but_wrong_change: %d"%(correct_changes, float(changens), noisy_but_wrong_changes, float(changens), nonoisy_but_wrong_changes))
    return f1


if __name__=='__main__':
  
    #preprocess.build_vocab()
    #preproce:q`ss.convert_to_id()
    #preprocess.prepare_word_vec()
    tf.app.flags.DEFINE_float("init_scale", 0.01, "initialize range")
    tf.app.flags.DEFINE_string("sent_pkl_file", "sen_pkl_file.csv", "sentence file")
    tf.app.flags.DEFINE_string("word_vec_pkl_file", "word_vector.csv", "word embedding file")
    tf.app.flags.DEFINE_float("lr", 0.5, "learning rate")
    tf.app.flags.DEFINE_float("lr_decay", 0.9, "lr decay")
    tf.app.flags.DEFINE_float("keep_prob", 0.5, "keep probability")
    tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size")
    tf.app.flags.DEFINE_integer("max_max_epoch", 60, "iteration")
    tf.app.flags.DEFINE_integer("max_epoch", 10, "iteration")
    tf.app.flags.DEFINE_integer("hidden_size", 200, "hidden size")
    tf.app.flags.DEFINE_float("max_grad_norm", 5.0, "max grad")
    tf.app.flags.DEFINE_boolean("max_pooling", True, "max pooling or mean pooling")
    tf.app.flags.DEFINE_integer("mslen", 95, "uniform sentence length")
    tf.app.flags.DEFINE_float("lam", 0.5, "lambda")
    tf.app.flags.DEFINE_float("startepoch", 100, "when to change label")
    tf.app.flags.DEFINE_float("noisy_per", 0.8, "noisy percent")
    tf.app.flags.DEFINE_string("reload_name", "", "which model to reload")
    tf.app.flags.DEFINE_string("save_dir", "/scr/leijh/ds_model/thebest/", "which model to reload")
    FLAGS = tf.app.flags.FLAGS
    #print(FLAGS.__flags)
    print(get_training_info())
    print(FLAGS.noisy_per)
    if FLAGS.noisy_per == 0:
        print('using clean data')
        sentences = pickle.load(open(FLAGS.sent_pkl_file))
    else:
        print('using %.1f noisy data!'%(FLAGS.noisy_per))
        sentences = pickle.load(open(FLAGS.sent_pkl_file+'%s'%(float(FLAGS.noisy_per))))
    #print(sentences[0])
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True 
    train_data_1, train_label_1, train_len_1, train_noisy_label, valid_data_1, valid_label_1, valid_len_1, valid_noisy_label, test_data, test_label, test_len, test_noisy_label = sentences
    for cross_label in range(1, 2):
        with tf.Graph().as_default(), tf.Session(config=_config) as session:
            initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
            #saver_path = saver.save(session, "/home/leijh/model1/extract.ckpt")
            #with tf.variable_scope("model", reuse=None, initializer=initializer):
            with tf.variable_scope("model", reuse=None):
                m = PTBModel(is_training=True, use_dropout=True, max_pooling = FLAGS.max_pooling, batch_size = FLAGS.batch_size)   
            #with tf.variable_scope("model", reuse=True, initializer=initializer):
            with tf.variable_scope("model", reuse=True):
                mvalid = PTBModel(is_training=False, use_dropout=False, max_pooling = FLAGS.max_pooling, batch_size = 160)
            #with tf.variable_scope("model", reuse=True, initializer=initializer):
            #with tf.variable_scope("model", reuse=True):
            #    mtest = PTBModel(is_training=False, use_dropout=False, max_pooling = FLAGS.max_pooling, batch_size = 1)
            saver=tf.train.Saver(max_to_keep=None)
        #ckpwith tf.variable_scope("model", reuse=None, initializer=initializer)t = tf.train.get_checkpoint_state("/home/leijh/model1/exp66")        
            if FLAGS.reload_name != "":
                model_path =  FLAGS.train_dir+FLAGS.reload_name
                print('restore from %s' % model_path)
                saver.restore(session, model_path)
            else:
                print('using fresh model!')
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
            m.assign_lr(session, FLAGS.lr)
            for i in range(FLAGS.max_max_epoch):
             
                
                  
            #  llr_decay = 0.5^(i-max_epoch)
                #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                #m.assign_lr(session, config.learning_rate * lr_decay) # learning rate
                #m.assign_lr(session, FLAGS.lr)
                print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(m.lr)))
                print("train-%d:"%(cross_label))
                train_per = run_epoch(session, m, train_data, train_noisy_label, train_label, train_len, i)
    
               #print("valid-%d:"%(cross_label))
                
                #saver_path = saver.save(session, "/scr/leijh/ds_model/save/epoch%d"%(i))
                valid_per = run_epoch(session, mvalid, valid_data, valid_noisy_label, valid_label, valid_len, i, test=True)
            #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                saver_path = saver.save(session, "/scr/leijh/ds_model/noisy0.8/epoch%d_%.3f"%(i, valid_per)) 
                if i>=FLAGS.max_epoch:
                    if valid_per<=maxvalid:
                        lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
                        m.assign_lr(session, session.run(m.lr) * FLAGS.lr_decay)
                else:
                    m.assign_lr(session, FLAGS.lr)
                if valid_per > maxvalid:
                    maxvalid = valid_per
                    #if valid_per > maxvalid: 
                #    maxvalid = valid_per
                #    print("test:")
                #    test_result = run_epoch(session, mtest, test_data, test_lable_s, test_lable_e, test_len, test_g, test = True)
            print(maxvalid)
