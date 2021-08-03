import getopt
import sys
from models.seqgan.Seqgan import Seqgan
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def set_gan():
    gan = Seqgan()
    gan.generate_num = 10000
    return gan



def main_train():
    gan = set_gan()
    gan.train()


if __name__ == '__main__':
    main_train()
