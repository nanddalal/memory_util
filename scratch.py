import memory_util
memory_util.vlog(1)

from utils.plotting.plotweb import *

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

BYTES_PER_MB = float(1024*1024)

in_shapes = [
    (1, 256, 256, 3),
    (2, 256, 256, 3),
    (4, 256, 256, 3),
    (8, 256, 256, 3),
    (16, 256, 256, 3),
    ]
kernel_size = 3
in_channels = 3
out_channels = 64

x = tf.placeholder(tf.float32, (None, None, None, in_channels))
y = slim.conv2d(x, out_channels, kernel_size)

gpu_options = tf.GPUOptions(
    allow_growth=True,
    )
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    gpu_options=gpu_options,
    )
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(
    # graph=graph,
    config=config,
    )

sess.run(tf.global_variables_initializer())

def run_and_analyze(in_shape):
  with memory_util.capture_stderr() as stderr:
    res = sess.run(y, feed_dict={x:np.random.randn(*in_shape)})

  print res.shape

  expected_mem = reduce(lambda i,j:i*j, in_shape) # inputs
  expected_mem += (kernel_size ** 2) + in_channels * out_channels # weights
  expected_mem += reduce(lambda i,j:i*j, res.shape) # outputs
  expected_mem *= 4 # 4 bytes per float

  peak_mem = memory_util.peak_memory(stderr)

  print 'expected mem usage (MB): ', expected_mem / BYTES_PER_MB
  print 'peak     mem usage (MB): ', peak_mem / BYTES_PER_MB
  print 'peak:expected mem ratio: ', peak_mem / expected_mem
  print memory_util.print_memory_timeline(stderr)
  memory_util.plot_memory_timeline(plt, stderr)

  import ipdb; ipdb.set_trace()

for in_shape in in_shapes:
  run_and_analyze(in_shape)
