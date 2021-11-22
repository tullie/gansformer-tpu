# Minimal script for generating images using pre-trained the GANformer
# Ignore all future warnings
# from warnings import simplefilter
# simplefilter(action = "ignore", category = FutureWarning)
# 
# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
# 
import sys
from training import misc
from training.misc import crop_max_rectangle as crop
import tflex
# 
# import dnnlib.tflib as tflib
# from pretrained_networks import load_networks # returns G, D, Gs
# # G: generator, D: discriminator, Gs: generator moving-average (higher quality images)
# 
# import tensorflow as tf

# def run(model, gpus, output_dir, images_num, truncation_psi, batch_size, ratio):
#     print("Initializing...")
#     # os.environ["CUDA_VISIBLE_DEVICES"] = gpus                   # Set GPUs
#     tflib.init_tf()                                             # Initialize TensorFlow
# 
#     sess = tf.get_default_session()
#     print(sess.list_devices())
# 
#     cores = tflex.get_cores()
#     tflex.set_override_cores(cores)
# 
#     print("Loading networks...")
#     G, D, Gs = load_networks(model)                             # Load pre-trained network
#     print("Printing layers...")
#     Gs.print_layers()                                           # Print network details
# 
#     print("Generate images...")
#     latents = np.random.randn(images_num, *Gs.input_shape[1:])  # Sample latent vectors
#     images = Gs.run(latents, truncation_psi = truncation_psi,   # Generate images
#         minibatch_size = batch_size, verbose = True)[0]
# 
#     print("Saving images...")
#     os.makedirs(output_dir, exist_ok = True)                    # Make output directory
#     pattern = "{}/Sample_{{:06d}}.png".format(output_dir)       # Output images pattern
#     for i, image in tqdm(list(enumerate(images))):              # Save images
#         crop(misc.to_pil(image), ratio).save(pattern.format(i))
# 
# def main():
#     parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
#     parser.add_argument("--model",              help = "Filename for a snapshot to resume (optional)", default = None, type = str)
#     parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
#     parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
#     parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
#     parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
#     parser.add_argument("--batch-size",         help = "Batch size for generating images (default: %(default)s)", default = 8, type = int)
#     parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
#     args = parser.parse_args()
#     run(**vars(args))
# 
# if __name__ == "__main__":
#     main()

import argparse
import os
import pathlib
from pathlib import Path
from pprint import pprint as pp

import numpy as np
import tensorflow as tf
import tqdm
from tqdm import tqdm

import dnnlib
import tflex
from dnnlib import EasyDict
from dnnlib import tflib
from training import misc
# from training.networks_stylegan2 import *

from io import BytesIO  # for Python 3


def rand_latent(n, seed=None):
    if seed is not None:
        if seed < 0:
            seed = 2*32 - seed
        np.random.seed(seed)
    result = np.random.randn(n, *G.input_shape[1:])
    if seed is not None:
        np.random.seed()
    return result


def tfinit():
    tflib.run(tf.global_variables_initializer())


def load_checkpoint(path, checkpoint_num=None):
    if checkpoint_num is None:
        ckpt = tf.train.latest_checkpoint(path)
        print(ckpt)
    else:
        ckpt = os.path.join(path, f'model.ckpt-{checkpoint_num}')
        print(ckpt)
        # gs://shapedai/clevr_model/model.ckpt-121856

    assert ckpt is not None
    print('Loading checkpoint ' + ckpt)
    saver.restore(sess, ckpt)
    return ckpt


def get_checkpoint(path):
    ckpt = tf.train.latest_checkpoint(path)
    return ckpt


def get_grid_size(n):
    gw = 1
    gh = 1
    i = 0
    while gw*gh < n:
        if i % 2 == 0:
            gw += 1
        else:
            gh += 1
        i += 1
    return (gw, gh)


def gen_images(latents, truncation_psi_val, outfile=None, display=False, labels=None, randomize_noise=False, is_validation=True, network=None, numpy=False):
    if outfile:
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    
    if network is None:
        network = Gs
    n = latents.shape[0]
    grid_size = get_grid_size(n)
    drange_net = [-1, 1]
    with tflex.device('/gpu:0'):
        result = network.run(latents, labels, truncation_psi_val=truncation_psi_val, is_validation=is_validation, randomize_noise=randomize_noise,
                             minibatch_size=sched.minibatch_gpu)
        result = result[:, 0:3, :, :]
        img = misc.convert_to_pil_image(
            misc.create_image_grid(result, grid_size), drange_net)
        if outfile is not None:
            img.save(outfile)
        if display:
            f = BytesIO()
            img.save(f, 'png')
    return result if numpy else img


def grab(save_dir, i, n=1, latents=None, **kwargs):
    if latents is None:
        latents = rand_latent(n, seed=i)
    gw, gh = get_grid_size(latents.shape[0])
    outfile = str(save_dir/str(i)) + '.png'
    return gen_images(latents, outfile=outfile, **kwargs)

# Conditional set: if property is not None, then assign d[name] := prop
# for every d in a set of dictionaries
def cset(dicts, name, prop):
    if not isinstance(dicts, list):
        dicts = [dicts]
    if prop is not None:
        for d in dicts:
            d[name] = prop

# Conditional set: if dict[name] is not populated from the command line, then assign dict[name] := prop
def nset(args, name, prop):
    flag = "--{}".format(name.replace("_", "-"))
    if flag not in sys.argv:
        args[name] = prop

# Conditional set: if dict[name] has its default value, then assign dict[name] := prop
def dset(d, name, prop, default):
    if d[name] == default:
        d[name] = prop

# Set network (generator or discriminator): model, loss and optimizer
def set_net(net, reg_interval):
    ret = EasyDict()
    ret.args  = EasyDict(func_name = "training.network.{}_GANformer".format(net[0])) # network options
    ret.loss_args = EasyDict(func_name = "training.loss.{}_loss".format(net[0]))      # loss options
    ret.opt_args  = EasyDict(beta1 = 0.0, beta2 = 0.99, epsilon = 1e-8)               # optimizer options
    ret.reg_interval = reg_interval
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleGAN2 TPU Generator')

    parser.add_argument('--model_dir', type=str, action='store',
                        help='Location of the checkpoint files')
    parser.add_argument('--save_dir', type=str, action='store',
                        help='Location of the directory to save images in')
    parser.add_argument('--truncation_psi', type=float, action='store',
                        help='Truncation psi (default: %(default)s)', default=0.65)
    parser.add_argument('--num_samples', type=int, action='store',
                        help='Number of samples to generate (default: %(default)s)',
                        default=1)
    parser.add_argument('--checkpoint_num', type=int, action='store',
                        help='The checkpoint to use to generate the images. The default is the latest checkpoint in model_dir', default=None)

    args = parser.parse_args()

    # environment variables
    # os.environ['TPU_NAME'] = 'dkgan-tpu'
    os.environ['NOISY'] = '1'

    # --- set resolution and label size here:
    label_size = 0
    resolution = 512
    fmap_base = (int(os.environ['FMAP_BASE']) if 'FMAP_BASE' in os.environ else 16) << 10
    num_channels = 3
    channel = 3
    count = 1
    grid_image_size = int(os.environ['GRID_SIZE']) if 'GRID_SIZE' in os.environ else 9
    # ------------------------
    print('------------------')
    print('Initializing model')
    print('------------------')

    # set up model
    dnnlib.tflib.init_tf()

    sess = tf.get_default_session()
    print(sess.list_devices())

    cores = tflex.get_cores()
    tflex.set_override_cores(cores)

    task = "clevr"

    model_args = EasyDict(**vars(args))

    nset(model_args, "recompile", None)
    nset(model_args, "mirror_augment", task in ["cityscapes", "ffhq"])

    nset(model_args, "transformer", True)
    nset(model_args, "components_num", {"clevr": 8}.get(task, 16))
    nset(model_args, "latent_size", {"clevr": 128}.get(task, 512))

    nset(model_args, "normalize", "layer")
    nset(model_args, "integration", "mul")
    nset(model_args, "kmeans", True)
    nset(model_args, "use_pos", True)
    nset(model_args, "mapping_ltnt2ltnt", task != "clevr")
    nset(model_args, "style", task != "clevr")

    nset(model_args, "g_arch", "resnet")
    nset(model_args, "d_arch", None)
    nset(model_args, "mapping_resnet", True)
    nset(model_args, "tanh", False)
    nset(model_args, "mapping_layersnum", None)
    nset(model_args, "mapping_lrmul", None)
    nset(model_args, "mapping_dim", 128)
    nset(model_args, "mapping_resnet", None)
    nset(model_args, "mapping_shared_dim", None)
    nset(model_args, "d_transformer", None)

    nset(model_args, "ltnt_gate", None)
    nset(model_args, "img_gate", None)
    nset(model_args, "iterative", None)
    nset(model_args, "kmeans_iters", None)

    nset(model_args, "num_heads", 1)
    nset(model_args, "pos_dim", None)
    nset(model_args, "pos_init", "uniform")
    nset(model_args, "pos_directions_num", 2)

    nset(model_args, "merge_layer", -1)
    nset(model_args, "merge_type", None)
    nset(model_args, "merge_same", None)

    nset(model_args, "g_start_res", 0)
    nset(model_args, "g_end_res", 8)

    nset(model_args, "d_start_res", 0)
    nset(model_args, "d_end_res", 8)

    nset(model_args, "g_ltnt2ltnt", None)
    nset(model_args, "g_img2img", 0)
    nset(model_args, "g_img2ltnt", None)

    nset(model_args, "d_ltnt2ltnt", None)
    nset(model_args, "d_img2img", 0)

    nset(model_args, "style_mixing", 0.9)
    nset(model_args, "component_mixing", 0.0)
    nset(model_args, "component_dropout", 0.0)
    nset(model_args, "attention_dropout", None)

    nset(model_args, "g_loss", "logistic_ns")
    nset(model_args, "g_reg_weight", 1.0)

    nset(model_args, "d_loss", "logistic")
    nset(model_args, "d_reg", "r1")

    gammas = {
        "ffhq": 10, 
        "cityscapes": 20, 
        "clevr": 40, 
        "bedrooms": 100
    }
    nset(model_args, "gamma", gammas.get(task, 10))        

    cG = set_net("G", reg_interval = 4)
    cD = set_net("D", reg_interval = 16)

    # Options for training loop.
    train = EasyDict(run_func_name = "training.training_loop.training_loop") # training loop options
    # Options for generator network.
    G_args = cG.args
    # Options for discriminator network.
    D_args = cD.args
    # Options for generator optimizer.
    G_opt = cG.opt_args
    # Options for discriminator optimizer.
    D_opt = cD.opt_args
    # Options for generator loss.
    G_loss = cG.loss_args
    # Options for discriminator loss.
    D_loss = cD.loss_args
    # Options for TrainingSchedule.
    sched = EasyDict()
    # Options for setup_snapshot_image_grid().
    grid = EasyDict(size='8k', layout='random')
    # Options for dnnlib.submit_run().
    sc = dnnlib.SubmitConfig()
    tf_config = {'rnd.np_random_seed': 1000}
    label_dtype = np.int64
    sched.minibatch_gpu = 1

    if model_args.components_num > 1:
        model_args.latent_size = int(model_args.latent_size / model_args.components_num)

    cD.args.latent_size = cG.args.latent_size = cG.args.dlatent_size = model_args.latent_size
    cset([cG.args, cD.args], "components_num", model_args.components_num)

    # Networks architecture
    cset(cG.args, "architecture", model_args.g_arch)
    cset(cD.args, "architecture", model_args.d_arch)
    cset(cG.args, "tanh", model_args.tanh)

    for arg in ["layersnum", "lrmul", "dim", "resnet", "shared_dim"]:
        field = "mapping_{}".format(arg)
        cset(cG.args, field, model_args[field])

    cset(cG.args, "transformer", model_args.transformer)
    cset(cD.args, "transformer", model_args.d_transformer)

    model_args.norm = model_args.normalize
    for arg in ["norm", "integration", "ltnt_gate", "img_gate", "iterative", "kmeans", 
                "kmeans_iters", "mapping_ltnt2ltnt"]:
        cset(cG.args, arg, model_args[arg])

    for arg in ["use_pos", "num_heads"]:
        cset([cG.args, cD.args], arg, model_args[arg])

    # Positional encoding
    for arg in ["dim", "init", "directions_num"]:
        field = "pos_{}".format(arg)
        cset([cG.args, cD.args], field, model_args[field])

    # k-GAN
    for arg in ["layer", "type", "same"]:
        field = "merge_{}".format(arg)
        cset(cG.args, field, model_args[field])
    cset([cG.args, train], "merge", False)

    # Attention
    for arg in ["start_res", "end_res", "ltnt2ltnt", "img2img"]: # , "local_attention"
        cset(cG.args, arg, model_args["g_{}".format(arg)])
        cset(cD.args, arg, model_args["d_{}".format(arg)])
    cset(cG.args, "img2ltnt", model_args.g_img2ltnt)
    # cset(cD.args, "ltnt2img", args.d_ltnt2img)

    # Mixing and dropout
    for arg in ["style_mixing", "component_mixing", "component_dropout", "attention_dropout"]:
        cset(cG.args, arg, model_args[arg])

    # Loss and regularization
    gloss_args = {
        "loss_type": "g_loss",
        "reg_weight": "g_reg_weight",
        # "pathreg": "pathreg",
    }
    dloss_args = {
        "loss_type": "d_loss",
        "reg_type": "d_reg",
        "gamma": "gamma"
    }
    for arg, cmd_arg in gloss_args.items():
        cset(cG.loss_args, arg, model_args[cmd_arg])
    for arg, cmd_arg in dloss_args.items():
        cset(cD.loss_args, arg, model_args[cmd_arg])

    if 'G' not in globals():
        with tflex.device('/gpu:0'):
            G = tflib.Network('G', num_channels=num_channels, resolution=resolution,
                              label_size=label_size, fmap_base=fmap_base, **G_args)
            G.print_layers()
            Gs, Gs_finalize = G.clone2('Gs')
            Gs_finalize()
            # D = tflib.Network('D', num_channels=num_channels, resolution=resolution,
            #                   label_size=label_size, fmap_base=fmap_base, **D_args)
            # D.print_layers()

    grid_size = (2, 2)
    gw, gh = grid_size
    gn = np.prod(grid_size)
    grid_latents = rand_latent(gn, seed=-1)
    grid_labels = np.zeros([gw * gh, label_size], dtype=label_dtype)

    tfinit()
    print('-----------------')
    print('Initialized model')
    print('-----------------')
    # ----------------------
    # Load checkpoint
    print('Loading checkpoint')
    saver = tf.train.Saver()

    # ------------------------
    model_ckpt = load_checkpoint(args.model_dir, args.checkpoint_num)
    model_name = model_ckpt.split('/')[4]
    model_num = model_ckpt.split('-')[-1]
    print('Loaded model', model_name)

    tflex.state.noisy = False
    # ---------------------
    
    save_dir = Path(args.save_dir)/model_num
    # generate samples
    for i in tqdm(list(range(args.num_samples))):
        grab(save_dir, i=i,
             truncation_psi_val=args.truncation_psi)  # modify seed
