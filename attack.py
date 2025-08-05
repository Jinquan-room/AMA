from __future__ import absolute_import, division, print_function

import os
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave
import tensorflow as tf

from tensorflow.contrib.slim.nets import inception_v3 as inception
slim = tf.contrib.slim

# 参数配置
tf.flags.DEFINE_string('checkpoint_path', '', 'Inception checkpoint.')
tf.flags.DEFINE_string('input_dir', '', 'Input images.')
tf.flags.DEFINE_string('output_dir', '', 'Output adversarial images.')
tf.flags.DEFINE_integer('image_height', 299, '')
tf.flags.DEFINE_integer('image_width', 299, '')
tf.flags.DEFINE_integer('batch_size', 10, '')
tf.flags.DEFINE_float('epsilon', 16.0, 'Max perturbation in L_inf norm.')
tf.flags.DEFINE_float('alpha', 1.0, 'Step size.')
tf.flags.DEFINE_integer('iterations', 10, 'Number of PGD steps.')
tf.flags.DEFINE_string('GPU_ID', '0', '')
FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID

# ---------- 加载图像 ----------
def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float32) / 255.0
            image = image * 2.0 - 1.0
        images[idx] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_shape[0]:
            yield filenames, images
            filenames, images, idx = [], np.zeros(batch_shape), 0
    if idx > 0:
        yield filenames, images

# ---------- 保存图像 ----------
def save_images(images, filenames, output_dir):
    for i, name in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, name), 'w') as f:
            imsave(f, ((images[i] + 1.0) * 0.5 * 255).astype(np.uint8), format='png')

# ---------- 注意力图提取 ----------
def compute_attention_map(sess, x_input, model_logits, feature_tensor, label_tensor):
    grads = tf.gradients(model_logits[:, label_tensor], feature_tensor)[0]  # ∂S/∂F
    grads_mean = tf.reduce_mean(grads, axis=[1, 2], keepdims=True)
    weighted_features = grads_mean * feature_tensor
    attention_map = tf.reduce_sum(tf.nn.relu(weighted_features), axis=3)  # ∑_c ReLU
    return sess.run(attention_map, feed_dict={x_input: x_batch})

# ---------- PGD 主循环 ----------
def pgd_attack(sess, x_input, logits, x_orig, y_target, A_ori, grad_tensor):
    eps = FLAGS.epsilon / 255.0
    alpha = FLAGS.alpha / 255.0
    x_adv = np.copy(x_orig)

    for i in range(FLAGS.iterations):
        loss_ce = tf.losses.softmax_cross_entropy(tf.one_hot(y_target, 1001), logits)
        A_adv = compute_attention_map(sess, x_input, logits, feature_tensor, y_target)
        loss_att = tf.reduce_mean(tf.square(A_adv - A_ori))
        total_loss = loss_ce + 0.1 * loss_att

        grad = tf.gradients(total_loss, x_input)[0]
        grad_val = sess.run(grad, feed_dict={x_input: x_adv})
        x_adv = x_adv + alpha * np.sign(grad_val)
        x_adv = np.clip(x_adv, x_orig - eps, x_orig + eps)
        x_adv = np.clip(x_adv, -1.0, 1.0)

    return x_adv

# ---------- 主函数 ----------
def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, endpoints = inception.inception_v3(x_input, num_classes=1001, is_training=False)
    feature_tensor = endpoints['Mixed_7c']  # 中间层特征图

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)

        for filenames, x_batch in load_images(FLAGS.input_dir, batch_shape):
            pred = sess.run(tf.argmax(logits, 1), feed_dict={x_input: x_batch})
            A_ori = compute_attention_map(sess, x_input, logits, feature_tensor, pred)
            x_adv = pgd_attack(sess, x_input, logits, x_batch, pred, A_ori, feature_tensor)
            save_images(x_adv, filenames, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()
