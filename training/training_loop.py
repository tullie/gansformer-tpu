# Training loop:
# 1. Sets up the environment and data
# 2. Builds the generator (g) and discriminator (d) networks
# 3. Manage the training process
# 4. Run periodic evaluations on specified metrics
# 5. Produces sample images over the course of training

# It supports training over data in TF records as produced by dataset_tool.py.
# Labels can optionally be provided although not essential
# If provided, image will be generated conditioned on a chosen label
import os
import glob
import numpy as np
import pathlib
import tensorflow as tf
import functools

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary, get_tpu_summary
import pretrained_networks
import tflex

from training import dataset as data
from training import misc
from training import visualize
from metrics import metric_base

# Data processing
# ----------------------------------------------------------------------------

# Just-in-time input image processing before feeding them to the networks
def process_reals(x, drange_data, drange_net, mirror_augment):
    with tf.name_scope("DynamicRange"):
        x = tf.cast(x, tf.float32)
        x.set_shape([None, 3, None, None])
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope("MirrorAugment"):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    return x

def read_data(data, name, shape, minibatch_gpu_in):
    # var = tf.Variable(name = name, trainable = False, initial_value = tf.zeros(shape))
    data_write = tf.concat([data, tf.zeros(shape)[minibatch_gpu_in:]], axis = 0)
    # data_fetch_op = tf.group(tf.assign(var, data_write), name="fetch_data")
    # data_read = var[:minibatch_gpu_in]
    data_read = data_write[:minibatch_gpu_in]
    return data_read #, data_fetch_op

# def set_shapes(batch_size, num_channels, resolution, label_size, images, labels, transpose_input=False):
#     """Statically set the batch_size dimension."""
#     if transpose_input:
#         shape = [resolution, resolution, num_channels, batch_size]
#         images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
#         images = tf.reshape(images, [-1])
#         labels.set_shape(labels.get_shape().merge_with(
#             tf.TensorShape([batch_size, label_size])))
#     else:
#         images.set_shape(images.get_shape().merge_with(
#             tf.TensorShape([batch_size, num_channels, resolution, resolution])))
#         labels.set_shape(labels.get_shape().merge_with(
#             tf.TensorShape([batch_size, label_size])))
#     return images, labels

# Scheduling and optimization
# ----------------------------------------------------------------------------

# Evaluate time-varying training parameters
def training_schedule(
    sched_args,
    cur_nimg,                      # The training length, measured in number of generated images
    dataset,                       # The dataset object for accessing the data
    lrate_rampup_kimg  = 0,        # Duration of learning rate ramp-up
    tick_kimg          = 8):       # Default interval of progress snapshots

    # Initialize scheduling dictionary
    s = dnnlib.EasyDict()

    # Set parameters
    s.kimg = cur_nimg / 1000.0
    s.tick_kimg = tick_kimg
    s.resolution = 2 ** dataset.resolution_log2

    for arg in ["G_lrate", "D_lrate", "minibatch_size", "minibatch_gpu"]:
        s[arg] = sched_args[arg]

    # Learning rate optional rampup
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    return s

# Build two optimizers a network cN for the loss and regularization terms
def set_optimizer(cN, lrate_in, minibatch_multiplier, lazy_regularization = True, clip = None):
    args = dict(cN.opt_args)
    args["minibatch_multiplier"] = minibatch_multiplier
    args["learning_rate"] = lrate_in
    if lazy_regularization:
        mb_ratio = cN.reg_interval / (cN.reg_interval + 1)
        args["learning_rate"] *= mb_ratio
        if "beta1" in args: args["beta1"] **= mb_ratio
        if "beta2" in args: args["beta2"] **= mb_ratio
    cN.opt = tflib.Optimizer(name = "Loss{}".format(cN.name), cross_shard=True, clip = clip, **args)
    cN.reg_opt = tflib.Optimizer(name = "Reg{}".format(cN.name), cross_shard=True, share = cN.opt, clip = clip, **args)

# Create optimization operations for computing and optimizing loss, gradient norm and regularization terms
def set_optimizer_ops(cN, lazy_regularization):
    cN.reg_norm = tf.constant(0.0)
    cN.trainables = cN.gpu.trainables

    if cN.reg is not None:
        if lazy_regularization:
            cN.reg_opt.register_gradients(tf.reduce_mean(cN.reg * cN.reg_interval), cN.trainables)
            cN.reg_norm = cN.reg_opt.norm
        else:
            cN.loss += cN.reg

    cN.opt.register_gradients(tf.reduce_mean(cN.loss), cN.trainables)
    cN.norm = cN.opt.norm

    cN.loss_op = tf.reduce_mean(cN.loss) if cN.loss is not None else tf.no_op()
    cN.regval_op = tf.reduce_mean(cN.reg) if cN.reg is not None else tf.no_op()

    reg_op = cN.reg_opt._shared_optimizers[''].minimize(cN.regval_op, var_list=cN.trainables) if cN.reg is not None else tf.no_op()
    train_op = cN.opt._shared_optimizers[''].minimize(cN.loss_op, var_list=cN.trainables)
    cN.ops = {"loss": cN.loss_op, "reg": cN.regval_op, "norm": cN.norm, "train_op": train_op, "reg_op": reg_op}


# Loading and logging
# ----------------------------------------------------------------------------

# Tracks exponential moving average: average, value -> new average
def emaAvg(avg, value, alpha = 0.995):
    if value is None:
        return avg
    if avg is None:
        return value
    return avg * alpha + value * (1 - alpha)

# Load networks from snapshot
def load_nets(resume_pkl, lG, lD, lGs, recompile):
    misc.log("Loading networks from %s..." % resume_pkl, "white")
    rG, rD, rGs = pretrained_networks.load_networks(resume_pkl)
    
    if recompile:
        misc.log("Copying nets...")
        lG.copy_vars_from(rG); lD.copy_vars_from(rD); lGs.copy_vars_from(rGs)
    else:
        lG, lD, lGs = rG, rD, rGs
    return lG, lD, lGs

# def get_input_fn(training_set, batch_size):
#     zz = training_set.get_minibatch_np(batch_size)
#     features = zz[0]
#     labels = zz[1]
#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#     def input_fn(params):
#         batch_size = params["batch_size"]
#         return dataset.repeat().batch(batch_size, drop_remainder=True)
#     return dataset, input_fn
# 
# def get_input_fn(training_set):
#     def input_fn(params):
#         import pdb; pdb.set_trace()
#         batch_size = params["batch_size"]
#         features, labels = training_set.get_minibatch_np(batch_size)
#         features = tf.cast(features, dtype=tf.float32)
#         labels = tf.cast(labels, dtype=tf.float32)
#         dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#         return dataset.repeat().batch(batch_size, drop_remainder=True)
#     return input_fn

def get_input_fn(load_training_set, num_cores, mirror_augment, drange_net):
    training_set = load_training_set()

    def input_fn(params):
        batch_size = params["batch_size"]

        num_channels = training_set.shape[0]
        resolution = training_set.shape[1]
        resolution_log2 = int(np.log2(resolution))
        label_size = training_set.label_size

        # def load_stylegan_tfrecord(tfr_files, to_float=False):
        #     dset = tf.data.Dataset.from_tensor_slices(tfr_files)
        #     dset = dset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=True))
        #     if training_set.label_file is not None:
        #         training_set._tf_labels_var, training_set._tf_labels_init = tflib.create_var_with_large_initial_value2(training_set._np_labels, name='labels_var', trainable=False)
        #         with tf.control_dependencies([training_set._tf_labels_init]):
        #             training_set._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(training_set._tf_labels_var)
        #     else:
        #         training_set._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(training_set._np_labels)
        #     dset = dset.map(data.TFRecordDataset.parse_tfrecord_tf_float if to_float else data.TFRecordDataset.parse_tfrecord_tf, num_parallel_calls=2)
        #     dset = tf.data.Dataset.zip((dset, training_set._tf_labels_dataset))
        #     return dset

        def dataset_parser_dynamic(reals, labels):
            reals = process_reals(reals, training_set.dynamic_range, drange_net, mirror_augment)
            # reals = read_data(reals, "reals", [batch_size] + training_set.shape, batch_size)
            # labels = read_data(labels, "labels", [batch_size, training_set.label_size], batch_size)
            return reals, labels

        training_set.configure(batch_size)
        
        features, labels = training_set.get_minibatch_np(batch_size)
        features = tf.cast(features, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        features, labels = dataset_parser_dynamic(features, labels)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.repeat().batch(batch_size, drop_remainder=True)

    # def input_fn(params):
    #     batch_size = params["batch_size"]
    #     num_channels = training_set.shape[0]
    #     resolution = training_set.shape[1]
    #     resolution_log2 = int(np.log2(resolution))
    #     label_size = training_set.label_size
    #     features, labels = training_set.get_minibatch_tf()
    #     features.set_shape((batch_size, num_channels, resolution, resolution))
    #     labels.set_shape((batch_size, label_size))
    #     features = tf.cast(features, dtype=tf.float32)
    #     labels = tf.cast(labels, dtype=tf.float32)
    #     dset = tf.data.Dataset.from_tensor_slices((features, labels))
    #     dset = dset.repeat().batch(batch_size, drop_remainder=True)
    #     return dset

    return input_fn, training_set

# Training Loop
# ----------------------------------------------------------------------------
# 1. Sets up the environment and data
# 2. Builds the generator (g) and discriminator (d) networks
# 3. Manage the training process
# 4. Run periodic evaluations on specified metrics
# 5. Produces sample images over the course of training

def training_loop(
    # Configurations
    cG = {}, cD = {},                   # Generator and Discriminator command-line arguments
    dataset_args            = {},       # dataset.load_dataset() options
    sched_args              = {},       # train.TrainingSchedule options
    vis_args                = {},       # vis.eval options
    grid_args               = {},       # train.setup_snapshot_img_grid() options
    metric_arg_list         = [],       # MetricGroup Options
    tf_config               = {},       # tflib.init_tf() options
    eval                    = False,    # Evaluation mode
    train                   = False,    # Training mode
    # Data
    data_dir                = None,     # Directory to load datasets from
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images
    mirror_augment          = False,    # Enable mirror augmentation?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks
    ratio                   = 1.0,      # Image height/width ratio in the dataset
    # Optimization
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters
    lazy_regularization     = False,     # Perform regularization as a separate training step?
    smoothing_kimg          = 10.0,     # Half-life of the running average of generator weights
    clip                    = None,     # Clip gradients threshold
    # Resumption
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning
                                        # Affects reporting and training schedule
    resume_time             = 0.0,      # Assumed wallclock time at the beginning, affects reporting
    recompile               = False,    # Recompile network from source code (otherwise loads from snapshot)
    # Logging
    summarize               = True,     # Create TensorBoard summaries
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    img_snapshot_ticks      = 3,        # How often to save image snapshots? None = disable
    network_snapshot_ticks  = 3,        # How often to save network snapshots? None = only save networks-final.pkl
    last_snapshots          = 10,       # Maximal number of prior snapshots to save
    eval_images_num         = 50000,    # Sample size for the metrics
    printname               = "",       # Experiment name for logging
    # Architecture
    merge                   = False):   # Generate several images and then merge them

    def load_training_set(**kws):
        return data.load_dataset(data_dir=dnnlib.convert_path(data_dir), verbose=True, **dataset_args, **kws)

    cG.name, cD.name = "g", "d"
    num_gpus = dnnlib.submit_config.num_gpus
    input_fn, dataset = get_input_fn(
        load_training_set,
        num_gpus,
        mirror_augment=mirror_augment,
        drange_net=drange_net,
    )

    sched = training_schedule(sched_args, cur_nimg = total_kimg * 1000, dataset = dataset)
    feed_dict = {
        "lrate_in_g": sched.G_lrate,
        "lrate_in_d": sched.D_lrate,
        "minibatch_size_in": sched.minibatch_size,
        "minibatch_gpu_in": sched.minibatch_gpu,
        "step": sched.kimg
    }

    def model_fn(features, labels, mode, params):
        nonlocal cG, cD, sched_args, lazy_regularization, smoothing_kimg
        assert mode == tf.estimator.ModeKeys.TRAIN

        G = tflib.Network("G", num_channels = dataset.shape[0], resolution = dataset.shape[1], 
            label_size = dataset.label_size, **cG.args)
        D = tflib.Network("D", num_channels = dataset.shape[0], resolution = dataset.shape[1], 
            label_size = dataset.label_size, **cD.args)

        G.print_layers()
        D.print_layers()

        Gs, Gs_finalize = G.clone2('Gs')
        reals_read = features
        labels_read = labels

        minibatch_gpu_in = feed_dict['minibatch_gpu_in']
        minibatch_size_in = feed_dict['minibatch_size_in']

        # Set optimizers
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        for cN, lr in [(cG, feed_dict["lrate_in_g"]), (cD, feed_dict["lrate_in_d"])]:
            set_optimizer(cN, lr, minibatch_multiplier, lazy_regularization, clip)

        # Create GPU-specific shadow copies of G and D
        for cN, N in [(cG, G), (cD, D)]:
            cN.gpu = N

        # Evaluate loss functions
        with tf.name_scope("G_loss"):
            cG.loss, cG.reg = dnnlib.util.call_func_by_name(G = cG.gpu, D = cD.gpu, dataset = dataset,
                reals = reals_read, minibatch_size = minibatch_gpu_in, **cG.loss_args)

        with tf.name_scope("D_loss"):
            cD.loss, cD.reg = dnnlib.util.call_func_by_name(G = cG.gpu, D = cD.gpu, dataset = dataset,
                reals = reals_read, labels = labels_read, minibatch_size = minibatch_gpu_in, **cD.loss_args)

        for cN in [cG, cD]:
            set_optimizer_ops(cN, lazy_regularization)

        inc_global_step = tf.assign_add(tf.train.get_or_create_global_step(), minibatch_gpu_in, name="inc_global_step")
        Gs_beta = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), smoothing_kimg * 1000.0) if smoothing_kimg > 0.0 else 0.0
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta = Gs_beta)
        loss = cG.ops["loss"] + cD.ops["loss"]
        if cG.reg is not None: loss += cG.ops["reg"]
        if cD.reg is not None: loss += cD.ops["reg"]

        with tf.control_dependencies([inc_global_step]):
            with tf.control_dependencies([cG.ops["train_op"]]):
                with tf.control_dependencies([cD.ops["train_op"]]):
                    with tf.control_dependencies([cG.ops["reg_op"]]):
                        with tf.control_dependencies([cD.ops["reg_op"]]):
                            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                                train_op = tf.group(Gs_update_op, name='train_op')

        print("Creating TPU estimator spec...")

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            host_call = get_tpu_summary().get_host_call(),
            loss=loss,
            train_op=train_op)

    tpu_cluster_resolver = tflex.get_tpu_resolver()
    model_dir=os.environ['MODEL_DIR']
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        save_checkpoints_secs=600 // 5,
        keep_checkpoint_max=10,
        keep_checkpoint_every_n_hours=1,
        cluster=tpu_cluster_resolver,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=256))

    training_steps = 2048 * 20480
    batch_size = feed_dict["minibatch_size_in"]
    estimator = tf.contrib.tpu.TPUEstimator(
        config=run_config,
        use_tpu=True,
        model_fn=model_fn,
        train_batch_size=batch_size,
    )

    print('Training...')
    estimator.train(input_fn, steps=training_steps)

    assert True == False

    # Initialize dnnlib and TensorFlow
    # tflib.init_tf(tf_config)
    # num_gpus = dnnlib.submit_config.num_gpus
    # cG.name, cD.name = "g", "d"

    # Load dataset, configure training scheduler and metrics object
    # dataset = data.load_dataset(data_dir = dnnlib.convert_path(data_dir), verbose = True, **dataset_args)
    # metrics = metric_base.MetricGroup(metric_arg_list)

    # Construct or load networks
    # with tflex.device("/gpu:0"):
    #     no_op = tf.no_op()
    #     G, D, Gs = None, None, None
    #     if resume_pkl is None or recompile:
    #         misc.log("Constructing networks...", "white")
    #         G = tflib.Network("G", num_channels = dataset.shape[0], resolution = dataset.shape[1], 
    #             label_size = dataset.label_size, **cG.args)
    #         D = tflib.Network("D", num_channels = dataset.shape[0], resolution = dataset.shape[1], 
    #             label_size = dataset.label_size, **cD.args)
    #         Gs = G.clone("Gs")
    #     if resume_pkl is not None:
    #         G, D, Gs = load_nets(resume_pkl, G, D, Gs, recompile)

    # G.print_layers()
    # D.print_layers()

    # Train/Evaluate/Visualize
    # Labels are optional but not essential
    # grid_size, grid_reals, grid_labels = misc.setup_snapshot_img_grid(dataset, **grid_args)
    # misc.save_img_grid(grid_reals, dnnlib.make_run_dir_path("reals.png"), drange = dataset.dynamic_range, grid_size = grid_size)
    # grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])

    # if eval:
    #     # Save a snapshot of the current network to evaluate
    #     pkl = dnnlib.make_run_dir_path("network-eval-snapshot-%06d.pkl" % resume_kimg)
    #     misc.save_pkl((G, D, Gs), pkl, remove = False)

    #     # Quantitative evaluation
    #     misc.log("Run evaluation...")
    #     metric = metrics.run(pkl, num_imgs = eval_images_num, run_dir = dnnlib.make_run_dir_path(),
    #         data_dir = dnnlib.convert_path(data_dir), num_gpus = num_gpus, ratio = ratio, 
    #         tf_config = tf_config, eval_mod = True, mirror_augment = mirror_augment)  

    #     # Qualitative evaluation
    #     misc.log("Produce visualizations...")
    #     visualize.eval(Gs, dataset, batch_size = sched.minibatch_gpu,
    #         drange_net = drange_net, ratio = ratio, **vis_args)

    # if not train:
    #     dataset.close()
    #     exit()

    # Setup training inputs
    # misc.log("Building TensorFlow graph...", "white")
    # with tf.name_scope("Inputs"), tflex.device("/cpu:0"):
    #     lrate_in_g           = tf.placeholder(tf.float32, name = "lrate_in_g", shape = [])
    #     lrate_in_d           = tf.placeholder(tf.float32, name = "lrate_in_d", shape = [])
    #     step                 = tf.placeholder(tf.int32, name = "step", shape = [])
    #     minibatch_size_in    = tf.placeholder(tf.int32, name = "minibatch_size_in", shape=[])
    #     minibatch_gpu_in     = tf.placeholder(tf.int32, name = "minibatch_gpu_in", shape=[])
    #     minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
    #     beta                 = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), 
    #                             smoothing_kimg * 1000.0) if smoothing_kimg > 0.0 else 0.0

    # Set optimizers
    # for cN, lr in [(cG, lrate_in_g), (cD, lrate_in_d)]:
    #     set_optimizer(cN, lr, minibatch_multiplier, lazy_regularization, clip)

    # Build training graph for each GPU
    # data_fetch_ops = []
    # for gpu in range(num_gpus):
    #     with tf.name_scope("GPU%d" % gpu), tflex.device("/gpu:%d" % gpu):

    #         # Create GPU-specific shadow copies of G and D
    #         for cN, N in [(cG, G), (cD, D)]:
    #             cN.gpu = N if gpu == 0 else N.clone(N.name + "_shadow")
    #         Gs_gpu = Gs if gpu == 0 else Gs.clone(Gs.name + "_shadow")

    #         # Fetch training data via temporary variables
    #         with tf.name_scope("DataFetch"):
    #             reals, labels = dataset.get_minibatch_tf()
    #             reals = process_reals(reals, dataset.dynamic_range, drange_net, mirror_augment)
    #             reals, reals_fetch = read_data(reals, "reals",
    #                 [sched.minibatch_gpu] + dataset.shape, minibatch_gpu_in)
    #             labels, labels_fetch = read_data(labels, "labels",
    #                 [sched.minibatch_gpu, dataset.label_size], minibatch_gpu_in)
    #             data_fetch_ops += [reals_fetch, labels_fetch]

    #         # Evaluate loss functions
    #         with tf.name_scope("G_loss"):
    #             cG.loss, cG.reg = dnnlib.util.call_func_by_name(G = cG.gpu, D = cD.gpu, dataset = dataset,
    #                 reals = reals, minibatch_size = minibatch_gpu_in, **cG.loss_args)

    #         with tf.name_scope("D_loss"):
    #             cD.loss, cD.reg = dnnlib.util.call_func_by_name(G = cG.gpu, D = cD.gpu, dataset = dataset,
    #                 reals = reals, labels = labels, minibatch_size = minibatch_gpu_in, **cD.loss_args)

    #         for cN in [cG, cD]:
    #             set_optimizer_ops(cN, lazy_regularization, no_op)

    # Setup training ops
    # data_fetch_op = tf.group(*data_fetch_ops)
    # for cN in [cG, cD]:
    #     cN.train_op = cN.opt.apply_updates()
    #     cN.reg_op = cN.reg_opt.apply_updates(allow_no_op = True)
    # Gs_update_op = Gs.setup_as_moving_average_of(G, beta = beta)

    # Finalize graph
    # with tflex.device("/gpu:0"):
    #     try:
    #         peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
    #     except tf.errors.NotFoundError:
    #         peak_gpu_mem_op = tf.constant(0)
    # tflib.init_uninitialized_vars()

    # Tensorboard summaries
    # if summarize:
    #     misc.log("Initializing logs...", "white")
    #     summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    #     if save_tf_graph:
    #         summary_log.add_graph(tf.get_default_graph())
    #     if save_weight_histograms:
    #         G.setup_weight_histograms(); D.setup_weight_histograms()

    # Initialize training
    # misc.log("Training for %d kimg..." % total_kimg, "white")
    # dnnlib.RunContext.get().update("", cur_epoch = resume_kimg, max_epoch = total_kimg)
    # maintenance_time = dnnlib.RunContext.get().get_last_update_interval()

    # cur_tick, running_mb_counter = -1, 0
    # cur_nimg = int(resume_kimg * 1000)
    # tick_start_nimg = cur_nimg
    # for cN in [cG, cD]:
    #     cN.lossvals_agg = {k: None for k in ["loss", "reg", "norm", "reg_norm"]}
    #     cN.opt.reset_optimizer_state()

    # Training loop
    # while cur_nimg < total_kimg * 1000:
    #     if dnnlib.RunContext.get().should_stop():
    #         break

    #     # Choose training parameters and configure training ops
    #     sched = training_schedule(sched_args, cur_nimg = cur_nimg, dataset = dataset)
    #     assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
    #     dataset.configure(sched.minibatch_gpu)

    #     # Run training ops
    #     feed_dict = {
    #         lrate_in_g: sched.G_lrate,
    #         lrate_in_d: sched.D_lrate,
    #         minibatch_size_in: sched.minibatch_size,
    #         minibatch_gpu_in: sched.minibatch_gpu,
    #         step: sched.kimg
    #     }

    #     # Several iterations before updating training parameters
    #     for _repeat in range(minibatch_repeats):
    #         rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
    #         for cN in [cG, cD]:
    #             cN.run_reg = lazy_regularization and (running_mb_counter % cN.reg_interval == 0)
    #         cur_nimg += sched.minibatch_size
    #         running_mb_counter += 1

    #         for cN in [cG, cD]:
    #             cN.lossvals = {k: None for k in ["loss", "reg", "norm", "reg_norm"]}

    #         # Gradient accumulation
    #         for _round in rounds:
    #             cG.lossvals.update(tflib.run([cG.train_op, cG.ops], feed_dict)[1])
    #             if cG.run_reg:
    #                 _, cG.lossvals["reg_norm"] = tflib.run([cG.reg_op, cG.reg_norm], feed_dict)

    #             tflib.run(data_fetch_op, feed_dict)

    #             cD.lossvals.update(tflib.run([cD.train_op, cD.ops], feed_dict)[1])
    #             if cD.run_reg:
    #                 _, cD.lossvals["reg_norm"] = tflib.run([cD.reg_op, cD.reg_norm], feed_dict)

    #         tflib.run([Gs_update_op], feed_dict)

    #         # Track loss statistics
    #         for cN in [cG, cD]:
    #             for k in cN.lossvals_agg:
    #                 cN.lossvals_agg[k] = emaAvg(cN.lossvals_agg[k], cN.lossvals[k])

        # Perform maintenance tasks once per tick
        # done = (cur_nimg >= total_kimg * 1000)
        # if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
        #     cur_tick += 1
        #     tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
        #     tick_start_nimg = cur_nimg
        #     tick_time = dnnlib.RunContext.get().get_time_since_last_update()
        #     total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

        #     # Report progress
        #     print(("tick %s kimg %s   loss/reg: G (%s %s) D (%s %s)   grad norms: G (%s %s) D (%s %s)   " + 
        #            "time %s sec/kimg %s maxGPU %sGB %s") % (
        #         misc.bold("%-5d" % autosummary("Progress/tick", cur_tick)),
        #         misc.bcolored("{:>8.1f}".format(autosummary("Progress/kimg", cur_nimg / 1000.0)), "red"),
        #         misc.bcolored("{:>6.3f}".format(cG.lossvals_agg["loss"] or 0), "blue"),
        #         misc.bold( "{:>6.3f}".format(cG.lossvals_agg["reg"] or 0)),
        #         misc.bcolored("{:>6.3f}".format(cD.lossvals_agg["loss"] or 0), "blue"),
        #         misc.bold("{:>6.3f}".format(cD.lossvals_agg["reg"] or 0)),
        #         misc.cond_bcolored(cG.lossvals_agg["norm"], 20.0, "red"),
        #         misc.cond_bcolored(cG.lossvals_agg["reg_norm"], 20.0, "red"),
        #         misc.cond_bcolored(cD.lossvals_agg["norm"], 20.0, "red"),
        #         misc.cond_bcolored(cD.lossvals_agg["reg_norm"], 20.0, "red"),
        #         misc.bold("%-10s" % dnnlib.util.format_time(autosummary("Timing/total_sec", total_time))),
        #         "{:>7.2f}".format(autosummary("Timing/sec_per_kimg", tick_time / tick_kimg)),
        #         "{:>4.1f}".format(autosummary("Resources/peak_gpu_mem_gb", peak_gpu_mem_op.eval() / 2**30)),
        #         printname))

        #     autosummary("Timing/total_hours", total_time / (60.0 * 60.0))
        #     autosummary("Timing/total_days", total_time / (24.0 * 60.0 * 60.0))

        #     # Save snapshots
        #     if img_snapshot_ticks is not None and (cur_tick % img_snapshot_ticks == 0 or done):
        #         visualize.eval(Gs, dataset, batch_size = sched.minibatch_gpu, training = True,
        #             step = cur_nimg // 1000, grid_size = grid_size, latents = grid_latents, 
        #             labels = grid_labels, drange_net = drange_net, ratio = ratio, **vis_args)

        #     if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
        #         pkl = dnnlib.make_run_dir_path("network-snapshot-%06d.pkl" % (cur_nimg // 1000))
        #         misc.save_pkl((G, D, Gs), pkl, remove = False)

        #         if cur_tick % network_snapshot_ticks == 0 or done:
        #             metric = metrics.run(pkl, num_imgs = eval_images_num, run_dir = dnnlib.make_run_dir_path(),
        #                 data_dir = dnnlib.convert_path(data_dir), num_gpus = num_gpus, ratio = ratio, 
        #                 tf_config = tf_config, mirror_augment = mirror_augment)

        #         if last_snapshots > 0:
        #             misc.rm(sorted(glob.glob(dnnlib.make_run_dir_path("network*.pkl")))[:-last_snapshots])

        #     # Update summaries and RunContext
        #     if summarize:
        #         metrics.update_autosummaries()
        #         tflib.autosummary.save_summaries(summary_log, cur_nimg)

        #     dnnlib.RunContext.get().update(None, cur_epoch = cur_nimg // 1000, max_epoch = total_kimg)
        #     maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot
    # misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path("network-final.pkl"), remove = False)

    # All done
    # if summarize:
    #     summary_log.close()
    # dataset.close()
