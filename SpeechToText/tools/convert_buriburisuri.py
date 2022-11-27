'''
A tool to convet model from https://github.com/buriburisuri/speech-to-text-wavenet
'''
import shutil
import utils
import wavenet
import tensorflow as tf

flags = tf.compat.v1.app.flags
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('input_dir', 'model/buriburisuri', 'Directory buriburisuri model.')
flags.DEFINE_string('output_path', 'release/buriburisuri', 'Path to output model.')
FLAGS = flags.FLAGS


def main(_):
  utils.load(FLAGS.config_path)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  inputs = tf.compat.v1.placeholder(tf.float32, [1, None, utils.Data.num_channel])
  wavenet.bulid_wavenet(inputs, len(utils.Data.vocabulary), is_training=False)
  restore = utils.restore_from_pretrain(FLAGS.input_dir)
  saver = tf.compat.v1.train.Saver()
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(restore)
    saver.save(sess, FLAGS.output_path)
  shutil.copy(FLAGS.config_path, FLAGS.output_path+'.json')


if __name__ == '__main__':
  tf.compat.v1.app.run()
