# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import datetime
import pickle
import math
import time
import sys
import os
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.rnn import LSTMStateTuple

word2id = pickle.load(open('word2id.pkl','rb'))
id2word = dict((v,k) for k, v in word2id.items())
wordtovec = pickle.load(open('wordtovec.pkl','rb'))

word2vec_size = 250

BOS = 0
PAD = 1
UNK = 2
EOS = word2id["。"]

max_seq_len = 15

#Clip gradients to this norm
max_gradient_norm = 5.0
sampling_prob = 0.5
val_data_num = 10000

epoch_num = 1000
full_step = 5
lr = 0.001

layer_num = 2
hidden_units = 512
b_size = 50
dropout = 0.75

def trim_dupl(words):
    return [ word for i, word in enumerate(words) if i == 0 or words[i] != words[i-1] ]

def inv_logit_decay(n, k=40, min_clip=0.1):
    return max(k / (k + np.exp(n/k)), min_clip)


class dataset:
    def __init__(self,txt_path,istrain):
        if(istrain):
            print("loading training data from: " + txt_path)
            self.ques = []
            self.ans = []
            self.cnt = val_data_num
            self.val_cnt = 0
            self.val_start = 0
            line_num = 0
            
            with open(txt_path,'r') as file:
                tmp = '+++$+++'
                delete = False
                for l in file:
                    line_num += 1
                    if(line_num > 1500000):
                        break
                    l = l.strip('\n')
                    if(l == "+++$+++"):
                        tmp = l
                    elif (l != None):
                        if((tmp == "+++$+++") or delete):
                            tmp = (str(l)+" 。"+'\n').split()
                            tmp = [word2id[word] if word in word2id else UNK for word in tmp]
                            delete = ((tmp.count(UNK) > len(tmp)*0.15) or (len(tmp) > 10)) or (len(tmp) < 2)
                        else:
                            #tmp.insert(0,BOS)
                            self.ques.append(tmp)
                            tmp = (str(l)+" 。"+'\n').split()
                            tmp = [word2id[word] if word in word2id else UNK for word in tmp]
                            delete = ((tmp.count(UNK) > len(tmp)*0.15) or (len(tmp) > 10)) or (len(tmp) < 2)
                            if(delete):
                                self.ques.pop()
                            else:
                                self.ans.append(tmp)
                self.train_num = len(self.ques)

                self.ques = np.array(self.ques)
                self.ans = np.array(self.ans)
            print("number of training samples:{}".format(self.train_num))

        else:
            print("loading testing data from:" + str(txt_path))
            self.test = []
            self.test_cnt = 0

            in_f = open('input1.txt','w+')
            with open(txt_path,'r') as file:
                for l in file:
                    if(l != None and l != '+++$+++'):
                        tmp = l.split()
                        for word in tmp:
                            in_f.write(word)
                        in_f.write('\n')
                        tmp.append('。')
                        self.test.append([word2id[word] if word in word2id else UNK for word in tmp])
            in_f.close()

            print("number of testing samples:{}".format(len(self.test)))

    def shuffle(self):
        shu = np.arange(self.train_num - val_data_num) + val_data_num
        np.random.shuffle(shu)
        shu = np.insert(shu,np.zeros(val_data_num),np.arange(val_data_num))
        self.ques = self.ques[shu] 
        self.ans = self.ans[shu]

    def next_batch(self,batch_size,isval=False):
        if(isval):
            if self.val_cnt + batch_size > val_data_num-1:
                self.val_cnt = 0
            self.val_start = self.val_cnt
            self.val_cnt += batch_size
            get_start = self.val_start
        else:
            if self.cnt + batch_size > self.train_num - 1:
                self.shuffle()
                self.cnt = val_data_num
            self.start = self.cnt
            self.cnt += batch_size
            get_start = self.start

        batch_x = np.zeros((batch_size, max_seq_len))
        batch_y = np.zeros((batch_size, max_seq_len))
        batch_x_len = np.zeros((batch_size))
        batch_y_len = np.zeros((batch_size))
        tmp_x=[]
        tmp_y=[]
        
        for i in range(batch_size):
            tmp_x = self.ques[get_start+i]
            batch_x_len[i] = len(tmp_x)
            tmp_y = self.ans[get_start+i]
            batch_y_len[i] = len(tmp_y)

            tmp_x = np.pad(tmp_x,[0,max_seq_len-len(tmp_x)],'constant',constant_values=(0,PAD))
            tmp_y = np.pad(tmp_y,[0,max_seq_len-len(tmp_y)],'constant',constant_values=(0,PAD))

            batch_x[i,:] = tmp_x
            batch_y[i,:] = tmp_y

        return batch_x, batch_y, batch_x_len, batch_y_len

    def get_test_batch(self,batch_size):
        self.test_start = self.test_cnt
        hasnext = True
        
        if(self.test_cnt + batch_size > len(self.test)-1):
            n_input = len(self.test) - self.test_start
            self.test_cnt = 0
            hasnext = False
        else:
            self.test_cnt += batch_size
            n_input = batch_size

        batch_x = np.zeros((n_input,max_seq_len))
        batch_x_len = np.zeros(n_input)
        for i in range(n_input):
            tmp_x = self.test[self.test_start + i]
            batch_x_len[i] = len(tmp_x)
            tmp_x = np.pad(tmp_x,[0,max_seq_len-len(tmp_x)],'constant',constant_values=(0,PAD))
            batch_x[i,:] = tmp_x

        return batch_x, batch_x_len, hasnext

    def get_step_num(self,batch_size):
        return(int((self.train_num - val_data_num)/batch_size))

    def get_ref(self,samp,isques,isval=False):
        if(isval):
            if(isques):
                return(self.ques[self.val_start+samp])
            else:
                return(self.ans[self.val_start+samp])
        else:
            if(isques):
                return(self.ques[self.start+samp])
            else:
                return(self.ans[self.start+samp])

class model:
    def __init__(self,istest):
        self.istest = istest
        self.isval = False
        sampling = True
        attention = True

        self.enc_input = tf.placeholder(shape=(None,max_seq_len),dtype=tf.int32,name='encoder_input')
        self.source_length = tf.placeholder(shape=(None,),dtype=tf.int32,name='source_length')
        self.dec_target = tf.placeholder(shape=(None,max_seq_len),dtype=tf.int32,name='decoder_input')
        self.target_length = tf.placeholder(shape=(None,),dtype=tf.int32,name='target_length')
        self.sampling_prob = tf.placeholder(shape=(),dtype=tf.float32,name='sampling_probability')
        
        self.n_input = tf.shape(self.enc_input)[0]
        self.vocab_num = len(word2id)


        with tf.variable_scope('embeddings') as scope:
            #embedding matrix
            '''init_emb = np.zeros([self.vocab_num,word2vec_size])
            #BOS, PAD = zero vector
            init_emb[UNK] = np.random.uniform(-0.5,0.5,word2vec_size)
            for word in word2id:
                ind = word2id[word]
                vec = wordtovec[word]
                init_emb[ind] = vec
            init_emb = tf.convert_to_tensor(init_emb,dtype = np.float32)'''

            embeddings = tf.get_variable(
                name='embedding',
                shape=[self.vocab_num, word2vec_size])
            #embeddings = tf.get_variable(
            #    name='embedding',
            #    initializer=init_emb)

            enc_input_emb = tf.nn.embedding_lookup(embeddings,self.enc_input)

            #add BOS in front of groudtruth decoder input
            bos_vec = tf.zeros([self.n_input,1],dtype = tf.int32)
            dec_input = tf.concat([bos_vec,self.dec_target],axis = 1)
            target_emb = tf.nn.embedding_lookup(embeddings,dec_input)
            


        #input format: [batch size, max_seq_len, word2vec_size]
        with tf.variable_scope('encoder') as scope:
            enc_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_units),input_keep_prob=dropout) for _ in range(layer_num)])         
            enc_outputs, self.enc_state = tf.nn.dynamic_rnn(
                cell=enc_cell,
                sequence_length=self.source_length,
                inputs=enc_input_emb,
                dtype=tf.float32,time_major=False)
            '''self.enc_state = tuple([LSTMStateTuple(tf.concat([f_st.c, f_st.c], 1),
                                             tf.concat([b_st.h, b_st.h], 1))
                              for f_st, b_st in zip(encoder_state[0],
                                                    encoder_state[1])])'''
            
            scope.reuse_variables()


        with tf.variable_scope('decoder') as scope:
            start_token = tf.zeros([self.n_input],dtype = tf.int32)
            end_token = tf.Variable(EOS,dtype = tf.int32)
            if(self.istest or self.isval):
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding = embeddings,
                    start_tokens = start_token,
                    end_token = end_token)
            else:
                if(sampling):
                    #samp_prob = tf.Variable(samp_prob, dtype = tf.float32)

                    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                        inputs = target_emb,
                        sequence_length = self.target_length,
                        embedding = embeddings,
                        sampling_probability = self.sampling_prob)
                else:
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs = target_emb,
                        sequence_length = self.target_length)

            dec_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_units),input_keep_prob=dropout) for _ in range(layer_num)])

            if(attention):
                att_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=hidden_units,
                    memory=enc_outputs,
                    memory_sequence_length=self.source_length)
                dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=dec_cell,
                    attention_mechanism=att_mechanism,
                    attention_layer_size=hidden_units)
                self.enc_state = dec_cell.zero_state(
                    dtype=tf.float32,batch_size=self.n_input).clone(
                    cell_state=self.enc_state)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=helper,
                initial_state=self.enc_state,
                output_layer=Dense(self.vocab_num, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)))

            final_outputs, final_state, final_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
            self.logits_raw = final_outputs.rnn_output
            scope.reuse_variables()

        with tf.variable_scope("loss") as scope:
            if not (self.istest):
                pad_size = max_seq_len - tf.reduce_max(final_seq_len)
                self.logits = tf.pad(self.logits_raw, [[0,0],[0,pad_size],[0,0]])
                weights = tf.sequence_mask(
                    lengths=self.target_length,
                    maxlen=max_seq_len,
                    dtype=tf.float32,
                    name='weights')
                x_entropy_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self.logits,
                    targets=self.dec_target,
                    weights=weights)
                self.loss = tf.reduce_mean(x_entropy_loss)

                #if not (self.isval):
                    #clipped gradients
                params = tf.trainable_variables()
                global_step = tf.Variable(0,trainable = False)
                
                #learning_rate = tf.train.cosine_decay(lr,global_step,5000)
                self.optimizer = tf.train.AdamOptimizer()#.minimize(self.loss)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

                self.train_op = self.optimizer.apply_gradients(
                    zip(gradients, params),global_step = global_step)

            self.output_id = tf.argmax(self.logits_raw,2)
 
            scope.reuse_variables()

    def step(self, sess, inputs,
            source_length, targets=None, target_length=None, test_mode=False, prob=0):

        if self.istest:
            output = sess.run(self.output_id,
                            {self.enc_input : inputs,
                             self.source_length : source_length,
                             self.sampling_prob : prob})
            return output
        elif self.isval:
            loss, output = sess.run([self.loss,self.output_id],
                            {self.enc_input : inputs,
                             self.source_length : source_length,
                             self.dec_target : targets,
                             self.target_length : target_length,
                             self.sampling_prob : prob})
            return loss, output
        else:
            _, loss, output, lr_tmp = sess.run([self.train_op, self.loss, self.output_id, self.optimizer._lr_t],
                            {self.enc_input : inputs,
                             self.source_length : source_length,
                             self.dec_target : targets,
                             self.target_length : target_length,
                             self.sampling_prob: prob})
            return loss, output, lr_tmp

    def set_mode(self,test,val):
        self.istest = test
        self.isval = val


def main(args):

    resume = args.resume
    model_dir = 'save4/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    if(args.mode=='train'):
        mydataset = dataset(args.file,True)
        mydataset.shuffle()
        total_steps = int(mydataset.get_step_num(b_size)/full_step)
        print("total_steps{}".format(total_steps))
        bar_length = 20

        mymodel = model(False)
        saver = tf.train.Saver(tf.global_variables())
        loss = []
        validation_loss = 0
        step_num = 0
        start = time.time()
        

        with tf.Session() as sess:
            if resume==0:
                print("Begin training...")
                sess.run(tf.global_variables_initializer())
                loss_file = open('loss.txt','w')
                loss_file.close()
                val_loss_file = open('val_loss.txt','w')
                val_loss_file.close()

            else:
                print("Start training form epoch {}".format(resume))
                saver.restore(sess, model_dir+'model.ckpt-{}'.format(resume-1))


            for epoch in range(resume,epoch_num):
                mymodel.set_mode(False,False)

                print("Epoch:{}/{}".format(epoch+1,epoch_num))
                for step in range(total_steps):
                    step_num += 1
                    t = time.time()
                    inputs, targets, inputs_len, targets_len = mydataset.next_batch(b_size)
                    '''for sen in range(len(inputs)):
                        print([id2word[word] for word in inputs[sen]])
                        print([id2word[word] for word in targets[sen]])'''
                    prob = inv_logit_decay(epoch)
                    step_loss, output, learning_rate = mymodel.step(
                        sess,inputs,inputs_len,targets,targets_len,prob)
                    remain_time = round((time.time()-t)*(total_steps-step))
                    progress = round(bar_length*(step/total_steps))
                    text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f "%(
                            '='*(progress-1)+'>'+'.'*(bar_length-progress),
                            remain_time,
                            step_loss
                            )
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    loss.append(step_loss)

                    #take last batch as sample
                    
                    if(step==total_steps-1):
                        print("start output:")
                        for samp in range(3):
                            print("Sample_{}".format(samp))
                            for ind in mydataset.get_ref(samp,True):
                                if(print_text(ind)): break
                            print()

                            print("  prediction:")
                            preds = []
                            for ind in output[samp]:
                                if ind in id2word:
                                    preds.append(id2word[ind])
                                else:
                                    preds.append("UNK")
                            preds = trim_dupl(preds)
                            print(preds)

                            print("  reference:")
                            for ind in mydataset.get_ref(samp,False):
                                if(print_text(ind)): break
                            print()
                        #start validation
                        print("validation data:")
                        mymodel.set_mode(False,True)
                        tmp_val_loss = []
                        val_output = []
                        for i in range(int(val_data_num/(b_size*5))):
                            val_input,val_target,val_input_len,val_target_len = mydataset.next_batch(b_size,isval=True)
                            val_loss, val_output = mymodel.step(
                                sess,val_input,val_input_len,val_target,val_target_len,test_mode = True)
                            tmp_val_loss.append(val_loss)
                        validation_loss = np.mean(tmp_val_loss)
                        print("validation loss:{}".format(validation_loss))
                        
                        print("start output:")
                        for samp in range(3):
                            print("Sample_{}".format(samp))
                            for ind in mydataset.get_ref(samp,True,True):
                                if(print_text(ind)): break
                            print()

                            print("  prediction:")
                            for ind in val_output[samp]:
                                if ind in id2word:
                                    print(id2word[ind],end=' ')
                                else:
                                    print("UNK",end = ' ')
                            print()

                            print("  reference:")
                            for ind in mydataset.get_ref(samp,False,True):
                                if(print_text(ind)): break
                            print()

                saver.save(sess, model_dir+'model.ckpt',global_step=epoch)

                
                with open('loss.txt','a') as file:
                    for r in loss:
                        file.write(str(r)+'\n')
                with open('val_loss.txt','a') as file:
                    file.write(str(validation_loss)+'\n')
                loss = []
                validation_loss = []
                print("time", str(datetime.timedelta(seconds=time.time() - start)))


    elif(args.mode=='test'):
        with tf.Session() as sess:
            testset = dataset(args.file,False)

            mymodel = model(True)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, model_dir+'model.ckpt-{}'.format(resume-1))

            mymodel.set_mode(True,False)
            hasnext=True
            
            out_f = open('output1.txt','a')
            batch_num = 0

            while(hasnext):
                batch_num += 1
                text = "\rtesting {} batch...".format(batch_num)
                sys.stdout.write(text)
                sys.stdout.flush()
                test_input,test_input_len,hasnext = testset.get_test_batch(b_size)
                '''for i in range(len(test_input)):
                    print([id2word[ind] for ind in test_input[i]])'''

                test_output = mymodel.step(sess,test_input,test_input_len,test_mode = True)
                
                for samp in range(len(test_output)):
                    if(test_output[samp] == None).all():
                        out_f.write('。')
                    else:
                        for ind in test_output[samp]:
                            #print(ind)
                            if(ind == EOS):
                                break
                            elif(ind!=UNK and ind!=PAD):
                                out_f.write(id2word[ind])
                    out_f.write('\n')
            out_f.close()
    else:
        print("unknown mode")
        return

def print_text(ind):
    if (ind==EOS):
        return True
    elif ind in id2word:
        print(id2word[ind],end=' ')
    else:
        print("UNK",end = ' ')
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help="parser txt file",type=str, required=True)
    parser.add_argument('-m','--mode',help="train or test",type=str,default="train")
    parser.add_argument('-r','--resume',help='resume training',type=int,default=0)
    args = parser.parse_args()

    main(args)
