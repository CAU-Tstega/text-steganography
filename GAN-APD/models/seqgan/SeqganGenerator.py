import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
import collections
import random
import copy
import Huffman_Encoding
from utils.utils import *


class Generator(object):
    def __init__(self, num_vocabulary, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token, dropout_keep_prob,
                 learning_rate=0.001, reward_gamma=0.95):
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, 
                                        dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.dropout_keep_prob = dropout_keep_prob
        

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([
                                            self.num_vocabulary, 
                                            self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  
            self.g_output_unit = self.create_output_unit(self.g_params)  

        self.x = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                 self.sequence_length])  
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                         self.sequence_length])  
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(
                                            self.g_embeddings, self.x),
                                            perm=[1, 0, 2])  

        # Initial states
        #self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        #self.h0 = tf.stack([self.h0, self.h0])

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, 
                                             size=self.sequence_length,
                                             dynamic_size=False, 
                                             infer_shape=True)

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, 
                                             size=self.sequence_length,
                                             dynamic_size=False, 
                                             infer_shape=True)



        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            cell_output, state = self.g_recurrent_unit(x_t, h_tm1)  
            o_t = self.g_output_unit(cell_output) 
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), 
                                            [self.batch_size]), tf.int32)

            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token) 
            gen_o = gen_o.write(i, tf.reduce_sum(
                                   tf.multiply(tf.one_hot(next_token, 
                                               self.num_vocabulary, 1.0, 0.0),
                                               tf.nn.softmax(o_t)), 1)) 

            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, state, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, 
                                              self.start_token), 
                                              self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack() 
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            cell_output, state = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(cell_output)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, state, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(
                       self.g_embeddings, 
                       self.start_token),
                       self.h0, g_predictions))

        self.g_predictions = tf.transpose(self.g_predictions.stack(),
                                          perm=[1, 0, 2]) 
        # pretraining loss
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), 
                       self.num_vocabulary, 1.0, 0.0) * \
					   tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, 
                              [-1, self.num_vocabulary]), 1e-20, 1.0))) / \
                              (self.sequence_length * self.batch_size)

        # training updates
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.g_params = tf.trainable_variables()
        self.pretrain_grad, _ = tf.clip_by_global_norm(
                                      tf.gradients(self.pretrain_loss, 
                                                   self.g_params), 
                                      self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(
                                       zip(self.pretrain_grad, self.g_params))

        ##############################################
        #  Unsupervised Training
        ##############################################
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), 
                           self.num_vocabulary, 1.0, 0.0) * \
                tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, 
                               [-1, self.num_vocabulary]), 1e-20, 1.0)), 1) * \
                       tf.reshape(self.rewards, [-1]))

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, 
                                                             self.g_params), 
                                                self.grad_clip)
												
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))
                                          
        # one-step
        self.input_data = tf.placeholder(tf.int32, 
                                         shape=[self.batch_size,])

        x_t_debug = tf.nn.embedding_lookup(self.g_embeddings, self.input_data)
        self.cell_output, self.h_t_debug = self.g_recurrent_unit(x_t_debug, 
                                                                 self.h0) 
        o_t = self.g_output_unit(self.cell_output)
        self.prob = tf.nn.softmax(o_t)
        self.log_prob = tf.log(self.prob)
        self.next_token = tf.cast(tf.reshape(tf.multinomial(self.log_prob, 1), 
                                  [self.batch_size]), tf.int32)

    def VLC_embed(self, sess, embed_bit, bit_stream):
        bit_index = 0
        word_num = 0
        bit_num = embed_bit
        secret_bits_num = 0
        init = sess.run(self.h0)
        feed = {self.h0:init, self.input_data:[0] * self.batch_size,
                self.keep_prob:1.0}
        state, log_prob, token = sess.run([self.h_t_debug, self.log_prob,
                                           self.next_token], feed)

        outputs = token.reshape(-1, 1)

        for i in range(self.sequence_length-1):
            feed = {self.h0: init, self.input_data: [0] * self.batch_size,
                    self.keep_prob: 1.0}
            state, log_prob, token = sess.run([self.h_t_debug, self.log_prob,
                                     self.next_token], feed)

            outputs = token.reshape(-1, 1)
            
            for i in range(self.sequence_length-1):
                feed = {self.h0: state, self.input_data:token, 
                        self.keep_prob:1.0}
                state, prob, token = sess.run([self.h_t_debug, self.prob,
                                     self.next_token], feed)

                temp = []
                for i in range(self.batch_size):
                    p = prob[i].reshape(-1)
                    prob_sort = sorted(p)
                    prob_sort.reverse()
                    word_prob = [prob_sort[n] for n in range(2**int(bit_num))]
                    p = p.tolist()
                    words_prob = [(p.index(word_prob[n]), word_prob[n])
                                  for n in range(2**int(bit_num))]
                    nodes = Huffman_Encoding.createNodes(
                                [item[1] for item in words_prob])
                    root = Huffman_Encoding.createHuffmanTree(nodes)
                    codes = Huffman_Encoding.huffmanEncoding(nodes, root)

                    for j in range(2**int(bit_num)):
                        if bit_stream[bit_index:bit_index+j+1] in codes:
                            code_index = codes.index(
                                         bit_stream[bit_index:bit_index+j+1])
                            gen = words_prob[code_index][0]
                            if gen==7592:
                                break
                            bit_index = bit_index+j+1
                            secret_bits_num += \
                                      len(bit_stream[bit_index:bit_index+j+1])
                            word_num += 1
                            break
                    temp.append(gen)
	            token = np.array(temp)
                outputs = np.hstack((outputs, token.reshape(-1, 1)))
                payload = float(secret_bits_num) / float(word_num)
            return outputs, payload




    def APD_embed(self, sess, bits):
        # init state
        init = sess.run(self.h0)
        # first word
        feed = {self.h0:init, self.input_data:[0] * self.batch_size, 
                self.keep_prob: 1.0}
        state, log_prob, token = sess.run([self.h_t_debug, self.log_prob, 
                                           self.next_token], feed)
        outputs = token.reshape(-1,1)
        probability = log_prob.reshape(64,-1)

        embed_bit = 0

        for _ in range(self.sequence_length-1):
            feed = {self.h0: state, self.input_data:token, self.keep_prob: 1.0}
            state, prob, token = sess.run([self.h_t_debug, self.prob, 
                                           self.next_token], feed)
            prob = prob.reshape(64,-1)
            thread_prob = 0.7
            prob_ = prob / prob[[list(range(64)),token]].reshape(-1,1)
            temp_p1 = prob[[list(range(64)),token]].reshape(-1,1)
            temp_p2 = prob[[list(range(64)),token]].reshape(-1,1)
            temp_p1[temp_p1 < thread_prob] = 1e+100
            temp_p2[temp_p2 > thread_prob] = 1e+100
            p1 = prob / temp_p1
            p2 = prob / temp_p2

            p1_up = [2, 10, 10]
            p1_down = [1, 0.09, 0.048]
            p2_up = [10, 50, 50]
            p2_down = [1, 0.49, 0.15]
            mask1 = np.logical_and(p1 < p1_up[bits-1], p1 >= p1_down[bits-1])
            mask2 = np.logical_and(p2 < p2_up[bits-1], p2 >= p2_down[bits-1])
            mask = mask1 | mask2
            # mask = np.logical_and(prob_ < 50, prob_ >= 1/50)
            #mask = prob_  == 1
            tmp = []
            for i in range(64):
                index = np.where(mask[i].reshape(1,-1))[1]
                try:
                    random = np.random.choice(index,1)
                except ValueError:
                    tmp.append(random)
                    continue
                tmp.append(random[0])
                embed_bit += int(np.log2(len(index)))
            token = np.array(tmp).reshape(64,)
            outputs = np.hstack((outputs, token.reshape(-1,1)))
            probability = np.hstack((probability,prob_.reshape(64,-1)))
        return outputs, probability, embed_bit


    def generate(self, sess):
        outputs = sess.run(self.gen_x, feed_dict={self.keep_prob: 1.0})
        return outputs

    def pretrain_step(self, sess, x):
        #run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], 
                  feed_dict={self.x: x, self.keep_prob: self.dropout_keep_prob})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        def get_cell():
            #self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                                       self.hidden_dim, 
                                       forget_bias=0.0, 
                                       state_is_tuple=False)

            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                                               lstm_cell, 
                                               output_keep_prob=self.keep_prob
                                               )
            return lstm_cell
        self.cell = tf.contrib.rnn.MultiRNNCell([get_cell() for _ in range(3)], 
                                                 state_is_tuple=True)
        self.h0 = self.cell.zero_state(self.batch_size, tf.float32)
        def unit(x, hidden_memory_tm1):
            (cell_output, state) = self.cell.__call__(x, hidden_memory_tm1) 
            return (cell_output, state)

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, 
                                                self.num_vocabulary]))
        self.bo = tf.Variable(self.init_matrix([self.num_vocabulary]))
        params.extend([self.Wo, self.bo])

        def unit(cell_output):            
            logits = tf.matmul(cell_output, self.Wo) + self.bo
            return logits
            
        return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:

    def set_similarity(self, valid_examples=None, pca=True):
        if valid_examples == None:
            if pca:
                valid_examples = np.array(range(20))
            else:
                valid_examples = np.array(range(self.num_vocabulary))
        self.valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.g_embeddings), 1, 
                                          keep_dims=True))
        self.normalized_embeddings = self.g_embeddings / self.norm
        # PCA
        if self.num_vocabulary >= 20 and pca == True:
            emb = tf.matmul(self.normalized_embeddings, 
                            tf.transpose(self.normalized_embeddings))
            s, u, v = tf.svd(emb)
            u_r = tf.strided_slice(u, begin=[0, 0], end=[20, 
                                   self.num_vocabulary], strides=[1, 1])
            self.normalized_embeddings = tf.matmul(u_r, self.normalized_embeddings)
        self.valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, 
                                    tf.transpose(self.normalized_embeddings))
