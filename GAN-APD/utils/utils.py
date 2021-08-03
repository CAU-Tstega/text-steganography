import numpy as np
import tensorflow as tf


def generate_VLC(sess, trainable_model, batch_size, embed_bit, generated_num,
				 bit_stream, output_file=None, get_code=True):
	# Generate Samples
	generated_samples = []
	BPW = 0
	for _ in range(int(generated_num / batch_size)):
		output, payload = trainable_model.VLC_embed(sess, embed_bit, bit_stream)
		BPW += payload
		generated_samples.extend(output)
		with open('save/payload.txt', 'a') as f:
			f.write('VLC True_payload: '+str(BPW/int(generated_num/batch_size))\
					+ '\n\n')

		codes, prob = list(), list()
		if output_file is not None:
			with open(output_file, 'w') as fout:
				for poem in generated_samples:
					buffer = ' '.join([str(x) for x in poem]) + '\n'
					fout.write(buffer)
					if get_code:
						codes.append(poem)
			return np.array(codes)



def generate_samples(sess, trainable_model, batch_size, generated_num,
                     output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes
				 

def generate_APD(sess, trainable_model, batch_size, bits, generated_num, 
                       output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    ppl = []
    embed_bits = 0
    for _ in range(int(generated_num / batch_size)):
        output, propability, embed_bit = trainable_model.APD_embed(sess, bits) 
        embed_bits += embed_bit
        #output = trainable_model.generate(sess)
        generated_samples.extend(output)
        ppl.extend(propability)
    print ('embed_bits: ', embed_bits)
    codes, prob = list(), list()
    if output_file is not None:
        #"""
        with open(output_file + "ppl", 'w') as fout:
            for poem in ppl:
                poem = poem.reshape(20,-1)
                for item in poem:
                    tmp = sorted(item, reverse=True)
                    buffer = ' '.join([str(x) for x in tmp[:20]]) + '\n'
                    fout.write(buffer)
        #"""
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)


def loadWord2Vec(dictionary):
    import gensim
    print ("using pre-trained word embeddind start...")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                     "./data/GoogleNews-vectors-negative300.bin",
                     binary = True, unicode_errors="ignore")
    word2vec_dict = {}
    word_embedding_2dlist = [[]]*(len(dictionary)+1)
    bound = np.sqrt(6.0) / np.sqrt(len(dictionary))
    count_exit = 0
    count_not_exit = 0
    for word in dictionary:
        embedding = None
        try:
            embedding = word2vec_model.wv[word[0]]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[int(word[1])] = embedding
            count_exit = count_exit+1
        else:
            word_embedding_2dlist[int(word[1])] = np.random.uniform(-bound,bound,300)
            count_not_exit = count_not_exit+1
    word_embedding_2dlist[-1] = np.random.uniform(-bound,bound,300)
    word_embedding = np.array(word_embedding_2dlist)
    print ("using pre-trained word embeddind ended ...")
    return word_embedding

def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    # sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
