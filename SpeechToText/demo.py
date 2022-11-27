import tensorflow as tf
import utils
import wavenet
import glog
import os

flags = tf.compat.v1.app.flags
flags.DEFINE_string('input_path', 'data/demo.wav', 'path to wav file.')
flags.DEFINE_string('evalute_path', 'model/v28/buriburisuri', 'Path to directory holding a checkpoint.')
flags.DEFINE_string('ckpt_model', 'model/v28/buriburisuri', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS


def main(_):
  tf.compat.v1.disable_eager_execution()
  if not os.path.exists(FLAGS.ckpt_model + '.index'):
    glog.error('%s was not found.' % FLAGS.ckpt_model)
    return -1

  utils.load(FLAGS.evalute_path + '.json')
  vocabulary = tf.constant(utils.Data.vocabulary)
  inputs = tf.compat.v1.placeholder(tf.float32, [1, None, utils.Data.num_channel])
  sequence_length = tf.compat.v1.placeholder(tf.int32, [None])

  logits = wavenet.bulid_wavenet(inputs, len(utils.Data.vocabulary), is_training=False)
  decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(a=logits, perm=[1, 0, 2]), sequence_length=sequence_length)
  outputs = tf.gather(vocabulary, tf.sparse.to_dense(decodes[0]))
  saver = tf.compat.v1.train.Saver()
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, FLAGS.ckpt_model)
    wave = utils.read_wave(FLAGS.input_path)
    output = utils.cvt_np2string(sess.run(outputs, feed_dict={inputs: [wave], sequence_length: [wave.shape[0]]}))[0]
    glog.info('%s: %s.', FLAGS.input_path, output)
  return 0


if __name__ == '__main__':
  tf.compat.v1.app.run()
