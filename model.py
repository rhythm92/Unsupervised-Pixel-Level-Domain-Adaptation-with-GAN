import tensorflow as tf
from dataset import *
from util import *

class DigistPLDAGAN():
    """Pixel-Level Domain Adaptation GAN used in Digists Datasets"""

    def __init__(src_domain='SVNH', trg_domain='MNIST'):
        pass

    def load_data_set(name):
        """Load a specific data set with name of choices :
           MNIST, MNISTM, SHVN, USPS

        Args:
            name:   (string) name of the dataset

        Outs:
            tensor with dimensionality (None, w, h, 3)
        """
        if name.upper() == 'MNIST':
            return load_MNIST()
        elif name.upper() == 'MNIST-M':
            return load_MNIST_M()
        elif name.upper() == 'USPS':
            return load_USPS()
        elif name.upper() == 'SVHN':
            return load_SVNH()
        else:
            raise Exception("Unrecognized dataset.")

    def build_T(x):
        """Build the graph for the classifier for digits datasets

        Args:
            x:  (batch_size, w, h, c)   image tensor from G or from source domain

        Outs:
            y:  (batch_size, 10)        Probability/Scores for each of the classes
        """
        pass

    def build_G(x_s,z):
        """Build the graph for the generator as specified in the paper.

        Args:
            x_s: (batch_size, w, h, c)  image tensor from the source domain
            z:   (batch_size, N)        a vector of random noises

        Outs:
            x_f: (batch_size, w', h', c) fake image tensor near target domain
        """
        pass

    def build_D(x):
        """Build the graph for the discriminator of digist datasets

        Args:
            x:  (batch_size, w, h, c)   image tensors from G or target domain

        Outs:
            p:  (batch_size,)           prob. of [x] comes from G
        """
        pass

    def build_model():
        pass

    def train():
        pass

    def test():
        pass

