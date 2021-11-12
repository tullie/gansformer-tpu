# Minimal script for generating images using pre-trained the GANformer
# Ignore all future warnings
from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)

import os
import argparse
import numpy as np
from tqdm import tqdm

from training import misc
from training.misc import crop_max_rectangle as crop
import tflex

import dnnlib.tflib as tflib
from pretrained_networks import load_networks # returns G, D, Gs
# G: generator, D: discriminator, Gs: generator moving-average (higher quality images)

import tensorflow as tf

# try:
#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
#   os.environ["TPU_NAME"] = tpu.cluster_spec().as_dict()['worker'][0]
#   print('Running on TPU ', os.environ["TPU_NAME"])
# except ValueError:
#   raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

if 'COLAB_TPU_ADDR' in os.environ:
  os.environ['TPU_NAME'] = 'grpc://' + os.environ['COLAB_TPU_ADDR']


def run(model, gpus, output_dir, images_num, truncation_psi, batch_size, ratio):
    print("Initializing...")
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus                   # Set GPUs
    tflib.init_tf()                                             # Initialize TensorFlow

    sess = tf.get_default_session()
    print(sess.list_devices())

    cores = tflex.get_cores()
    tflex.set_override_cores(cores)

    print("Loading networks...")
    G, D, Gs = load_networks(model)                             # Load pre-trained network
    print("Printing layers...")
    Gs.print_layers()                                           # Print network details

    print("Generate images...")
    latents = np.random.randn(images_num, *Gs.input_shape[1:])  # Sample latent vectors
    images = Gs.run(latents, truncation_psi = truncation_psi,   # Generate images
        minibatch_size = batch_size, verbose = True)[0]

    print("Saving images...")
    os.makedirs(output_dir, exist_ok = True)                    # Make output directory
    pattern = "{}/Sample_{{:06d}}.png".format(output_dir)       # Output images pattern
    for i, image in tqdm(list(enumerate(images))):              # Save images
        crop(misc.to_pil(image), ratio).save(pattern.format(i))

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
    parser.add_argument("--model",              help = "Filename for a snapshot to resume (optional)", default = None, type = str)
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
    parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
    parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
    parser.add_argument("--batch-size",         help = "Batch size for generating images (default: %(default)s)", default = 8, type = int)
    parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
    args = parser.parse_args()
    run(**vars(args))

if __name__ == "__main__":
    main()
