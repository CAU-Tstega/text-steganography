import json
from time import time

from models.Gan import Gan
from models.seqgan.SeqganGenerator import Generator
from utils.text_process import *
from utils.utils import *
import pickle, os


class Generate(Gan,object):
    def __init__(self, oracle=None):
        super(Generate,self).__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 300
        self.hidden_dim = 800
        self.sequence_length = 20
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0
        self.dropout_keep_prob = 0.5

        self.embed_alg = "APD"
        # self.embed_alg = "VLC"
        
        self.train_data = 'data/movie-filter30-150k.txt'

        self.generator_file = 'save/generator.txt'
        self.oracle_file = 'save/oracle_file.txt'



  
    def init_generate(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = self.train_data
        _, self.vocab_size = text_precess(data_loc)
        print("sequence_length: ", str(self.sequence_length))
        print("vocab_size: ", str(self.vocab_size))
        generator = Generator(num_vocabulary=self.vocab_size,  
                              batch_size=self.batch_size,          
                              emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, 
                              sequence_length=self.sequence_length,
                              start_token=self.start_token, 
                              dropout_keep_prob=self.dropout_keep_prob)
        self.set_generator(generator)

        tokens = get_tokenlized(data_loc)
        self.word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(self.word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, 
                                       self.sequence_length))

        return word_index_dict, index_word_dict



    def generate_embedtext(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        
        wi_dict, iw_dict = self.init_generate(data_loc)

        saver = tf.train.Saver()

        def get_real_test_file(dicts=iw_dict):
            with open(self.generator_file, 'r') as f:
                codes = get_tokenlized(self.generator_file)
            
            with open(self.words_file, 'a') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dicts))

        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver([v for v in tf.trainable_variables()])

        model_path = 'model/model'
        saver.restore(self.sess, model_path)
        print 'Language Model loaded....'

        bit_stream = open('data/bit_stream.txt').readline()


        if self.embed_alg is 'VLC':
            for embed_bit in range(1, 4):
                print self.embed_alg + " " + str(embed_bit)+" bpw embedding..."
                generate_VLC(self.sess, self.generator, self.batch_size, \
                             embed_bit, self.generate_num, bit_stream, \
                             self.generator_file)
                self.words_file = 'save/VLC_' + str(embed_bit) + 'bpw.txt'
                get_real_test_file()
                print 'Embedding Done...'


        if self.embed_alg is 'APD':
            for embed_bit in range(1, 4):
                print self.embed_alg + " " + str(embed_bit) + " bpw embedding..."
                generate_APD(self.sess, self.generator, self.batch_size, \
                             embed_bit, self.generate_num, self.generator_file)
                self.words_file = 'save/APD_' + str(embed_bit) + 'bpw.txt'
                get_real_test_file()
                print 'Embedding Done...'

