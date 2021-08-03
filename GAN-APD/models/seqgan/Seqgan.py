import json
from time import time

from models.Gan import Gan
from models.seqgan.SeqganDataLoader import DataLoader, DisDataloader
from models.seqgan.SeqganDiscriminator import Discriminator
from models.seqgan.SeqganGenerator import Generator
from models.seqgan.SeqganReward import Reward
from utils.text_process import *
from utils.utils import *

class Seqgan(Gan,object):
    def __init__(self, oracle=None):
        super(Seqgan,self).__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 300
        self.hidden_dim = 800
        self.sequence_length = 20
        self.filter_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, \
                            100, 100, 160, 160]
        self.l2_reg_lambda = 0.2
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0
        self.num_topN = 4
        self.dropout_keep_prob = 0.5
        
        self.train_data = 'data/movie.txt'

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'


    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, \
                         self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, \
                                             self.generator_file)
        loss_ = 0
        num_batch = 3
        for _ in range(num_batch):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            loss,_ = self.sess.run([self.discriminator.d_loss, \
                                    self.discriminator.train_op], feed)
            loss_ += loss
        print(loss_/num_batch)
        if self.log_ is not None:
             self.log_.write('epochs, '+str(self.epoch))
             self.log_.write(str(loss_/num_batch))
             self.log_.write('\n')

  
    def init_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = self.train_data
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        # self.sequence_length = 20
        print("sequence_length: ", str(self.sequence_length))
        print("vocab_size: ", str(self.vocab_size))

        generator = Generator(num_topN=self.num_topN, 
                              num_vocabulary=self.vocab_size,  
                              batch_size=self.batch_size,          
                              emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, 
                              sequence_length=self.sequence_length,
                              start_token=self.start_token, 
                              dropout_keep_prob=self.dropout_keep_prob)

        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, 
                                      num_classes=2,  
                                      vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, 
                                      filter_sizes=self.filter_size, 
                                      num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)

        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, 
                                    seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, 
                                       seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, 
                             dis_loader=dis_dataloader, 
                             oracle_loader=oracle_dataloader)

        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, 
                          self.sequence_length))

        return word_index_dict, index_word_dict


    def train(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_trainng(data_loc)
        
        
        saver = tf.train.Saver()

        def get_test_file(epoch, dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file+str(epoch), 'w') as outfile: 
                outfile.write(code_to_text(codes=codes, dictionary=dict))
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 300
        self.adversarial_epoch_num = 200
        # self.log = open('experiment-log-seqgan-real.csv', 'w')
        # self.log_ = open('experiment-log-seqgan-real-discriminator.csv', 'w')
        self.gen_data_loader.create_batches(self.oracle_file)
	
       	
        saver_g = tf.train.Saver([v for v in tf.all_variables() \
                     if ('generator' in v.name or 'multi_rnn_cell' in v.name)])
        model_g_path = 'model/generator-300'
        saver_g.restore(self.sess, model_g_path)
        print ("generator model loaded done!")
        self.epoch = 300	
     
       
        # '''
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, 
                                   self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) +'\t loss:'+ str(loss) + \
                  '\t time:' + str(end - start))

            self.add_epoch()
            
            if self.epoch != 0 and self.epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, \
                                 self.generate_num, self.generator_file)
                get_test_file(epoch=self.epoch)
        
            if self.epoch != 0 and self.epoch % 20 == 0:
                saver_g = tf.train.Saver([v for v in tf.all_variables() \
                      if ('generator' in v.name or 'multi_rnn_cell' in v.name)])
                saver_g.save(self.sess,'model/generator-'+str(self.epoch))
        # '''
       
	        
        saver_g = tf.train.Saver([v for v in tf.all_variables() \
                      if ('generator' in v.name or 'multi_rnn_cell' in v.name)])
        model_g_path = 'model/generator-300'
        saver_g.restore(self.sess, model_g_path)
        print ("generator model loaded done!")
        

        # '''
        print('start pre-train discriminator:')
        for epoch in range(500):
            print('D_TRN epoch:' + str(epoch))
            self.train_discriminator()
        saver_d = tf.train.Saver([v for v in tf.all_variables() \
                                                 if 'discriminator' in v.name])
        saver_d.save(self.sess,'model/discriminator')
        # '''
        
        saver_d = tf.train.Saver([v for v in tf.all_variables() \
                                                 if 'discriminator' in v.name])
        model_d_path = 'model/discriminator'
        saver_d.restore(self.sess, model_d_path)
        print ("discriminator model loaded done!")
        
        
        
        self.epoch = 300 #temp
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 64, \
                                                 self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards,
                    self.generator.keep_prob: 1.0
                }
                loss, _ = self.sess.run([self.generator.g_loss, 
                                         self.generator.g_updates], 
                                         feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if self.epoch != 0 and self.epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, \
                                 self.generate_num, self.generator_file)
                get_test_file(epoch=self.epoch)
                saver_g = tf.train.Saver()
                saver_g.save(self.sess,'model/generator-'+str(self.epoch))

            self.reward.update_params()
            for _ in range(5):
                self.train_discriminator()
        saver.save(self.sess,'model/model')
      

