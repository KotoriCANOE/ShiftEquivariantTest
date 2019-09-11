import os
import numpy as np
import pandas as pd
import tensorflow as tf

# boolean argument for argparser
def bool_argument(argp, name, default):
    argp.add_argument('--' + name, dest=name, action='store_true')
    argp.add_argument('--no-' + name, dest=name, action='store_false')
    eval('argp.set_defaults({}={})'.format(name, 'True' if default else 'False'))

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# reset random seeds
def reset_random(seed=0):
    import random
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# setup tensorflow and return session
def create_session(graph=None, memory_fraction=1.0, debug=False):
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True,
        per_process_gpu_memory_fraction=memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(graph=graph, config=config)
    if debug:
        from tensorflow.python import debug as tfdbg
        sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    return sess

######

def logSigmoid(x, offset=1e-1):
    return 1 - np.log(x + offset) / np.log(offset)

def gray2color(gray, colormap, float_out=False):
    if gray.dtype == np.uint16:
        gray = np.float32(gray) * (1 / 65535)
    color = colormap(gray)
    if not float_out:
        color = np.uint8(color * 255 + 0.5)
    return color

def crop(src, height_offset, width_offset, height, width):
    return src[height_offset : height_offset + height,
        width_offset : width_offset + width]

def padCrop(src, height_offset, width_offset, mode='reflect'):
    if height_offset >= 0:
        crop_t = height_offset
        crop_b = None
        pad_h = (0, height_offset)
    elif height_offset < 0:
        crop_t = 0
        crop_b = height_offset
        pad_h = (-height_offset, 0)
    if width_offset >= 0:
        crop_l = width_offset
        crop_r = None
        pad_w = (0, width_offset)
    elif width_offset < 0:
        crop_l = 0
        crop_r = width_offset
        pad_w = (-width_offset, 0)
    dst = src
    dst = np.pad(dst, (pad_h, pad_w), mode=mode)
    dst = dst[crop_t : crop_b, crop_l : crop_r]
    return dst

######

def inference(args, sess, src):
    mod = args.length_mod
    # input
    height = src.shape[0]
    width = src.shape[1]
    pad_height = (height + mod - 1) // mod * mod - height
    pad_width = (width + mod - 1) // mod * mod - width
    pad_t = pad_height // 2
    pad_b = pad_height - pad_t
    pad_l = pad_width // 2
    pad_r = pad_width - pad_l
    src = np.pad(src, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'reflect')
    src = np.expand_dims(np.transpose(src, (2, 0, 1)), 0) # HWC => NCHW
    # run session
    fetch = 'Output:0'
    feed_dict = {'Input:0': src}
    dst = sess.run(fetch, feed_dict)
    # output (NCHW => HWC)
    dst = np.squeeze(dst, 0)
    dst = np.transpose(dst, (1, 2, 0))
    dst = dst[pad_t : -pad_b if pad_b > 0 else None, pad_l : -pad_r if pad_r > 0 else None]
    return dst

def process_shifts(args, sess, img):
    shift_range = args.shift_range
    sh = img.shape[0]
    sw = img.shape[1]
    dh = sh - shift_range
    dw = sw - shift_range
    ah = dh - shift_range
    aw = dw - shift_range
    # convert img to float32
    if img.dtype == np.uint8:
        img = np.float32(img) * (1 / 255)
    elif img.dtype == np.uint16:
        img = np.float32(img) * (1 / 65535)
    elif img.dtype != np.float32:
        img = np.float32(img)
    # loop over shift positions on 2 axis
    dst_ref = None
    shift_map = np.empty((shift_range, shift_range))
    for y in range(shift_range):
        for x in range(shift_range):
            # crop a sub-image for processing, as a method to perform shifts
            src = crop(img, y, x, dh, dw)
            # run the session to process the sub-image
            dst = inference(args, sess, src)
            # crop aligned sub-images for all shifts
            dst_align = crop(dst, shift_range - y, shift_range - x, ah, aw)
            # assign the reference result
            if y == 0 and x == 0:
                dst_ref = dst_align
            # calculate the MAD and store it to the shift map
            diff = np.mean(np.abs(dst_align - dst_ref))
            shift_map[y, x] = diff
    # return shift map
    return shift_map

def process(args, sess):
    from skimage import io
    from matplotlib import cm
    # get file list
    extensions = ['.jpeg', '.jpg', '.png', '.bmp']
    files = [f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in extensions]
    # statistics
    ofile_csv = os.path.join(args.output_dir,
        'statistics.{}.csv'.format(args.postfix))
    csv_columns = ['mean', 'stddev', 'min', 'median', 'max']
    csv_index = []
    csv_data = []
    # loop over files
    for fname in files:
        ifile = os.path.join(args.input_dir, fname)
        ofile = os.path.join(args.output_dir,
            '{}.{}'.format(os.path.splitext(fname)[0], args.postfix))
        ofile_img = ofile + '.png'
        # read image
        img = io.imread(ifile)
        # process
        shift_map = process_shifts(args, sess, img)
        # create heat map
        shift_heatmap = gray2color(logSigmoid(shift_map, 1e-4), cm.inferno)
        # write result
        io.imsave(ofile_img, shift_heatmap)
        # write shift map statistics
        csv_index.append('[{}]{}'.format(args.postfix, fname))
        csv_data.append([np.mean(shift_map), np.std(shift_map),
            np.min(shift_map), np.median(shift_map), np.max(shift_map)])
    # write statistics
    csv_df = pd.DataFrame(csv_data, index=csv_index, columns=csv_columns)
    with open(ofile_csv, 'w') as fd:
        csv_df.to_csv(fd)

def test(args):
    # arXiv 1509.09308
    # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # create save directory
    if os.path.exists(args.output_dir):
        eprint('Confirm removing {}\n[Y/n]'.format(args.output_dir))
        if input() != 'Y':
            import sys
            sys.exit()
        import shutil
        shutil.rmtree(args.output_dir)
        eprint('Removed: ' + args.output_dir)
    os.makedirs(args.output_dir)
    # initialize
    if args.random_seed is not None:
        reset_random(args.random_seed)
    # create graph
    graph = tf.Graph()
    # load model
    model_file = os.path.join(args.model_dir, 'model.pb')
    graph_def = tf.GraphDef()
    with open(model_file, 'rb') as fd:
        graph_def.ParseFromString(fd.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    # process
    with create_session(graph) as sess:
        process(args, sess)

######

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # parameters
    argp.add_argument('input_dir')
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--device', default='/gpu:0')
    argp.add_argument('--postfix', default='')
    argp.add_argument('--model-dir', default='./model{postfix}.tmp')
    argp.add_argument('--output-dir', default='./shift{postfix}.tmp')
    # process parameters
    argp.add_argument('--shift-range', type=int, default=32)
    argp.add_argument('--length-mod', type=int, default=16)
    # parse
    args = argp.parse_args(argv)
    args.model_dir = args.model_dir.format(postfix=args.postfix)
    args.output_dir = args.output_dir.format(postfix=args.postfix)
    # run testing
    test(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
