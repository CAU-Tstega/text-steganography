import getopt
import sys
from models.seqgan.generate import Generate
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def set_gan():
	Gan = Generate
	gan = Gan()
	gan.generate_num = 10000
	return gan



def main_generate():
	gan = set_gan()
	gan.generate_embedtext()


if __name__ == '__main__':
	main_generate()
