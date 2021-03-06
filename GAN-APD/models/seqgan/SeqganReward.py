import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Reward(object):
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate

        self.num_vocabulary = self.lstm.num_vocabulary
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate
        self.topN = self.lstm.topN

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        #####################################################################################################
        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                 self.sequence_length])  # sequence of tokens generated by generator
        self.given_num = tf.placeholder(tf.int32)
        self.random_indices = tf.placeholder(tf.int32, shape=(self.batch_size,), name="random_indices")

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),
                                            perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))
        #####################################################################################################

        #self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        #self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            cell_output, state = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, state, given_num, gen_x

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            cell_output, state = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(cell_output)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, state, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        def _g_recurrence_2_topN(i, x_t, h_tm1, given_num, gen_x):
            cell_output, state = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(cell_output)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            topN = tf.nn.top_k(log_prob, self.topN)
            topN_i = topN.indices
            row_indices = tf.range(self.batch_size)
            random_indices = tf.random_uniform(shape=(self.batch_size,), minval=0, maxval=self.topN, dtype=tf.int32, seed=None, name=None)
            full_indices = tf.stack([row_indices, random_indices], axis=1)
            next_token = tf.gather_nd(topN_i, full_indices)            
            # filter next_token. if the best token is the eof_token, we use it rather than the token coming from topN sample.
            top1 = tf.nn.top_k(log_prob, 1) #batch_size x 1 (values, indices)
            top1_i = top1.indices #batch_size x 1
            top1_i = tf.reshape(top1_i, shape=[self.batch_size,]) #shape=(self.batch_size,) 
            masked = tf.equal(top1_i, (self.num_vocabulary-1))
            next_token_normal = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            next_token = tf.where(masked, next_token_normal, next_token)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, state, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x_topN = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x_topN = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x_topN))

        self.gen_x_topN = self.gen_x_topN.stack()  # seq_length x batch_size
        self.gen_x_topN = tf.transpose(self.gen_x_topN, perm=[1, 0])  # batch_size x seq_length


    def get_reward(self, sess, input_x, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, len(input_x[0])):
                feed = {self.x: input_x, self.given_num: given_num, self.lstm.keep_prob: 1.0}
                samples = sess.run(self.gen_x, feed)
                feed = {discriminator.input_x: samples}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: input_x}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[(len(input_x[0])-1)] += ypred

        reward_res = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return reward_res

    # use topN sample
    def get_reward_topN(self, sess, input_x, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, len(input_x[0])):
                feed = {self.x: input_x, self.given_num: given_num, self.lstm.keep_prob: 1.0}
                samples = sess.run(self.gen_x, feed) #
                feed = {discriminator.input_x: samples}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: input_x}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[(len(input_x[0])-1)] += ypred

        reward_res = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return reward_res
        ##################### end

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.cell = self.lstm.cell
        self.h0 = self.cell.zero_state(self.batch_size, tf.float32)
        def unit(x, hidden_memory_tm1):
            (cell_output, state) = self.cell.__call__(x, hidden_memory_tm1) 
            return (cell_output, state)

        return unit

    def update_recurrent_unit(self):
        self.cell = self.lstm.cell       
        
        def unit(x, hidden_memory_tm1):
            (cell_output, state) = self.cell.__call__(x, hidden_memory_tm1) 
            return (cell_output, state)

        return unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(cell_output):            
            logits = tf.matmul(cell_output, self.Wo) + self.bo
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(cell_output):
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(cell_output, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits
        return unit

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()
