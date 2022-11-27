import os
import glog
import tensorflow as tf
import utils
import wavenet
import dataset

flags = tf.compat.v1.flags
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('dataset_path', 'data/v28/train.record', 'Filepath to train dataset record.')
flags.DEFINE_integer('batch_size', 8, 'Batch size of train.') #32
flags.DEFINE_integer('display', 100, 'Step to display loss.')
flags.DEFINE_integer('snapshot', 1000, 'Step to save model.')
flags.DEFINE_integer('device', 0, 'The device used to train.')
flags.DEFINE_string('pretrain_dir', 'pretrain', 'Directory to pretrain.')
flags.DEFINE_string('ckpt_path', 'model/v28/ckpt', 'Path to directory holding a checkpoint.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate of train.')
FLAGS = flags.FLAGS


def main(_):
  print('branis1')
  os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
  utils.load(FLAGS.config_path)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  train_dataset = dataset.create(FLAGS.dataset_path, FLAGS.batch_size, repeat=True)
  print('branis2')
  # bug tensorflow!!!  the  train_dataset[0].shape[0] != FLAGS.batch_size once in a while
  # waves = tf.reshape(tf.sparse.to_dense(train_dataset[0]), shape=[FLAGS.batch_size, -1, utils.Data.num_channel])
  waves = tf.sparse.to_dense(train_dataset[0])
  waves = tf.reshape(waves, [tf.shape(input=waves)[0], -1, utils.Data.num_channel])


  labels = tf.cast(train_dataset[1], tf.int32)
  sequence_length = tf.cast(train_dataset[2], tf.int32)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary), is_training=True)
  loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels, logits, sequence_length, time_major=False))

  vocabulary = tf.constant(utils.Data.vocabulary)
  decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(a=logits, perm=[1, 0, 2]), sequence_length=sequence_length)
  outputs = tf.gather(vocabulary, tf.sparse.to_dense(decodes[0]))
  labels = tf.gather(vocabulary, tf.sparse.to_dense(labels))

  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimize = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss=loss, global_step=global_step)

  save = tf.compat.v1.train.Saver(max_to_keep=1000)
  config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(train_dataset[-1])
    if os.path.exists(FLAGS.pretrain_dir) and len(os.listdir(FLAGS.pretrain_dir)) > 0:
      save.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrain_dir))
    ckpt_dir = os.path.split(FLAGS.ckpt_path)[0]
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    if len(os.listdir(ckpt_dir)) > 0:
      save.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    losses, tps, preds, poses = 0, 0, 0, 0
    while True:
      gp, ll, uid, ot, ls, _ = sess.run((global_step,  labels, train_dataset[3], outputs, loss, optimize))
      tp, pred, pos = utils.evalutes(utils.cvt_np2string(ot), utils.cvt_np2string(ll))
      tps += tp
      losses += ls
      preds += pred
      poses += pos
      if gp % FLAGS.display == 0:
        glog.info("Step %d: loss=%f, tp=%d, pos=%d, pred=%d, f1=%f." %
                  (gp, losses if gp == 0 else (losses / FLAGS.display), tps, preds, poses,
                   2 * tps / (preds + poses + 1e-10)))
        losses, tps, preds, poses = 0, 0, 0, 0
      if (gp+1) % FLAGS.snapshot == 0 and gp != 0:
        save.save(sess, FLAGS.ckpt_path, global_step=global_step)


if __name__ == '__main__':
  tf.compat.v1.app.run()
