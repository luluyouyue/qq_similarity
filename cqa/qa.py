# encoding=utf8
import argparse# encoding=utf8

import os
import tensorflow as tf
import numpy as np
from . import utils
from . import sentence_embedding
from .utils import str2bool

from .utils import UNK_ID, PAD
from functools import reduce


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=10,
# parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
                    help='Path to the training data directory.')
parser.add_argument('--buckets', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"), \
                    help='Path to the training data directory.')
parser.add_argument('--train_file', type=str, default="train_q", \
                    help='name of training file.')
parser.add_argument('--checkpointDir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model"),
                    help='Path to the checkpoint directory.')
parser.add_argument('--summaryDir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "summary"),
                    help='Path to the summary directory.')

parser.add_argument("--use_quora_data", type=str2bool, default='False',
                    help='if use_quora_data or not')
parser.add_argument("--use_qq_data", type=str2bool, default='False',
                    help='if use_qq_data or not')

parser.add_argument('--eval_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval"),
                    help='Path to the evaluation result directory.')

parser.add_argument("--tb_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "log/summary"),
                    help='path to summary log dir')

parser.add_argument("--do_test", type=str2bool, default='False',
                    help='train or test mode')
parser.add_argument("--test_file", type=str, default="qq_test_golden.txt.tokenized",
                    help='test file used in evaluation')
parser.add_argument("--epoch", type=int, default=200,
                    help='epoch size')
parser.add_argument('--neg_sample_num', type=int, default=64,
                    help='num of negative samples for each positive sample')

parser.add_argument('--embedding_path', type=str, default="vectors_with_no_eos.txt",
                    help='embedding path of word vector')
parser.add_argument('--embedding_size', type=int, default=200,
                    help='embedding size of word vector')
parser.add_argument('--sentence_len', type=int, default=40,
                    help='sentence len. sentence with length lesser will padd with <PAD>')

parser.add_argument('--vocab_path', type=str, default="vocab.txt",
                    help='embedding path of word vector')

parser.add_argument('--use_hidden', type=str2bool, default='False',
                    help='whether add hidden layers after convolution or not')
parser.add_argument('--hidden_layer_size', type=int, default=500,
                    help='hidden layer size')
parser.add_argument('--hidden_layer_num', type=int, default=3,
                    help='number hidden layer')

parser.add_argument('--filter_size', type=int, default=1000,
                    help='num of filters')

parser.add_argument("--num_keep_ckpts", type=int, default=5,
                    help='num of checkpoints to keep')

parser.add_argument("--with_softmax", type=str2bool, default='False',
                    help="only use softmax loss")
parser.add_argument("--num_classes", type=int, default=2,
                    help="num classes in softmax")

parser.add_argument("--with_margin_loss", type=str2bool, default='True',
                    help="use margin loss")
parser.add_argument('--margin', type=float, default=0.5,
                    help='margin betwwen similarity between similar q,a pair and unsimilar q,a pair')


FLAGS, _ = parser.parse_known_args()

class Mode(object):
    train = "train"
    test = "test"
    eval = "eval"


class PredictMode(object):
    softmax = 'softmax'
    cosine_sim = 'cosine_sim'


class CQA(object):
    """
    A community based question answering implementation.
    https://arxiv.org/pdf/1508.01585.pdf
    """
    def __init__(self, vocab_id, embedding, mode=Mode.train):
        # batch_size * sentence_len
        self.input_question = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sentence_len], name="input_question")  # defualt = 40
        self.input_answer = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sentence_len], name="input_answer")

        # self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        self.vocab_id = vocab_id
        self.batch_size = tf.shape(self.input_question)[0]

        # sentence encoding
        question_emb = sentence_embedding.cnn_encoder(self.input_question, FLAGS.sentence_len, embedding,
                                                      FLAGS.embedding_size,  # default=200
                                                      filter_size=FLAGS.filter_size,  # default=1000, number of filters
                                                      use_hidden=FLAGS.use_hidden,    # whether add hidden layers after convolution or not, default=false
                                                      hidden_layer_num=FLAGS.hidden_layer_num, # default=3
                                                      hidden_size=FLAGS.hidden_layer_size, # default=500
                                                      name_scope='cqa')
        answer_emb = sentence_embedding.cnn_encoder(self.input_answer, FLAGS.sentence_len, embedding,
                                                    FLAGS.embedding_size,
                                                    filter_size=FLAGS.filter_size,
                                                    use_hidden=FLAGS.use_hidden,
                                                    hidden_layer_num=FLAGS.hidden_layer_num,
                                                    hidden_size=FLAGS.hidden_layer_size,
                                                    name_scope='cqa',
                                                    reuse=True)

        self.cos_q_a = cos_q_a = tf.reduce_sum(tf.multiply(question_emb, answer_emb), axis=-1)/tf.clip_by_value(tf.norm(question_emb, axis=-1) * tf.norm(answer_emb, axis=-1), 1e-10, np.inf)
        self.global_step = self.global_step = tf.Variable(0, trainable=False)

        if FLAGS.with_softmax: # default='True'
            # softmax
            if FLAGS.use_hidden:
                softmax_w = tf.get_variable("softmax_w", [1 * FLAGS.hidden_layer_size + 1, FLAGS.num_classes], dtype=tf.float32, initializer=tf.random_normal_initializer())  # define W
            else:
                softmax_w = tf.get_variable("softmax_w", [1 * 2 * FLAGS.filter_size + 1, FLAGS.num_classes], dtype=tf.float32, initializer=tf.random_normal_initializer()) # define b

            softmax_b = tf.get_variable("softmax_b", [FLAGS.num_classes], initializer=tf.constant_initializer(0.0)) # num_classes, num classes in softmax 2

            cnn_outputs = tf.concat([
                                     # question_emb,
                                     # answer_emb,
                                     tf.expand_dims(cos_q_a, -1),
                                     tf.abs(question_emb -answer_emb),
                                     ], axis=-1)
            
            if mode != Mode.train:
                self.softmax_logits = tf.matmul(cnn_outputs, softmax_w) + softmax_b
                # print self.softmax_logits
                self.softmax = tf.nn.softmax(self.softmax_logits)

        if mode == Mode.train:
            self.loss = None
            assert FLAGS.with_margin_loss or FLAGS.with_softmax, "should at least use one task."

            self.input_wrong_answer = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sentence_len],
                                                     name="input_wrong_answer")
            wrong_answer_emb = sentence_embedding.cnn_encoder(self.input_wrong_answer, FLAGS.sentence_len, embedding,
                                                              FLAGS.embedding_size,
                                                              filter_size=FLAGS.filter_size,
                                                              use_hidden=FLAGS.use_hidden,
                                                              hidden_layer_num=FLAGS.hidden_layer_num,
                                                              hidden_size=FLAGS.hidden_layer_size,
                                                              name_scope='cqa',
                                                              reuse=True)

            # max margin loss
            # loss = max {0, m - cos(logit_q, logit_a) + cos(logit_q, logit_wa})
            q_wa = tf.reduce_sum(tf.multiply(question_emb, wrong_answer_emb), axis=-1)
            q_wa_norm = tf.multiply(tf.norm(wrong_answer_emb, axis=-1), tf.norm(question_emb, axis=-1))
            self.cos_q_wa = cos_q_wa = q_wa / tf.clip_by_value(q_wa_norm, 1e-10, np.inf)

            if FLAGS.with_margin_loss:
                self.margin_loss = margin_loss = tf.reduce_sum(tf.nn.relu(tf.fill(tf.shape(cos_q_wa), FLAGS.margin) + cos_q_wa - cos_q_a, name="margin_loss"))
                self.loss = margin_loss
                tf.summary.scalar('margin_loss', margin_loss)

            # todo more feature could be included here
            if FLAGS.with_softmax:
                # softmax
                neg_cnn_outputs = tf.concat([
                                            # question_emb,
                                            tf.expand_dims(cos_q_wa, -1),
                                            # wrong_answer_emb,
                                            tf.abs(question_emb - wrong_answer_emb)], axis=-1)

                cnn_outputs = tf.concat([cnn_outputs, neg_cnn_outputs], axis=0)
                self.labels = tf.concat([tf.fill([self.batch_size], 1), tf.fill([self.batch_size], 0)], axis=0)


                softmax_logits = tf.matmul(cnn_outputs, softmax_w) + softmax_b

                self.cross_entropy = cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_logits, labels=self.labels), axis=0)
                if self.loss is not None:
                    self.loss = self.loss + cross_entropy
                else:
                    self.loss = cross_entropy
                tf.summary.histogram("softmax_w", softmax_w)
                tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('loss', self.loss)

            tf.summary.histogram("neg_cosine_sim", self.cos_q_wa)
            tf.summary.histogram("pos_cosine_sim", self.cos_q_a)

            # optimizer
            self.update_multi_task = tf.train.AdamOptimizer().minimize(loss=self.loss, global_step=self.global_step)

            params = tf.trainable_variables()
            # gradients = tf.gradients(loss, params)
            # self.update_step = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

            # Print trainable variables
            print("# Trainable variables")
            for param in params:
                print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                              param.op.device))

        self.summaries = tf.summary.merge_all()
        # saver
        self.saver = tf.train.Saver()

    def create_or_load_model(self, sess, model_dir, force_load=False):
        full_model_path = tf.train.latest_checkpoint(model_dir)
        if full_model_path:
            print("load model from %s ..." % full_model_path)
            self.saver.restore(sess, full_model_path)
        else:
            if force_load:
                raise ValueError("%s does not contain any model file.")
            else:
                print("init model with fresh parameter...")
                sess.run(tf.global_variables_initializer())

    def train(self, sess, input_question, input_answer, input_wrong_answer):
        ided_input_question = self.to_padded_token_ids(input_question)
        ided_input_answer = self.to_padded_token_ids(input_answer)
        ided_input_wrong_answer = self.to_padded_token_ids(input_wrong_answer)

        return sess.run([
            self.cos_q_a,
            self.loss,
            self.update_multi_task,
            self.summaries,
        ], feed_dict={self.input_question: ided_input_question,
                      self.input_answer: ided_input_answer,
                      self.input_wrong_answer: ided_input_wrong_answer})

    def to_padded_token_ids(self, tokenizd_input):
        _tokenizd_input = np.asarray(tokenizd_input)
        raw_shape = _tokenizd_input.shape
        _tokenizd_input = np.reshape(_tokenizd_input, [-1, raw_shape[-1]])
        if raw_shape[-1] < FLAGS.sentence_len:
            _tokenizd_input = np.concatenate([_tokenizd_input,
                                              np.full([reduce(lambda x, y: x * y, raw_shape[:-1]), FLAGS.sentence_len - raw_shape[-1]], PAD)], axis=-1)
        else:
            _tokenizd_input = _tokenizd_input[:, :FLAGS.sentence_len]

        padded_shape = [i for i in raw_shape[:-1]]
        padded_shape.append(FLAGS.sentence_len)
        return np.reshape(
            [self.vocab_id.get(utils.to_unicode(x).strip(), UNK_ID) for x in np.reshape(_tokenizd_input, [-1])], padded_shape)

    def predict(self, sess, input_question, input_answer, predictMode=PredictMode.cosine_sim):
        """
        predict the similarity between input question and input answer
        :param sess:
        :param input_question: list of tokenized input question tokens
        :param input_answer: list of tokenized input question tokens
        :return:
        """
        ided_input_question = self.to_padded_token_ids([input_question])
        ided_input_answer = self.to_padded_token_ids([input_answer])
        if predictMode == PredictMode.cosine_sim:
            if not FLAGS.with_margin_loss:
                raise ValueError("model not configured with margin loss")
            # [batch_size]
            cos_sim = sess.run([
                self.cos_q_a
            ], feed_dict={
                self.input_question: ided_input_question,
                self.input_answer: ided_input_answer
            })
            return cos_sim[0]
        elif predictMode == PredictMode.softmax:
            if not FLAGS.with_softmax:
                raise ValueError("model not configured with softmax")
            # [batch_size, num_class]
            softmax_logits, softmax = sess.run([
                self.softmax_logits,
                self.softmax
            ], feed_dict={
                self.input_question: ided_input_question,
                self.input_answer: ided_input_answer
            })

            print(softmax_logits, softmax, softmax[0][1])
            return softmax[0][1]

        else:
            raise ValueError("Unsupport mode: %s" % predictMode)

if __name__ == '__main__':
    print('training file = ', os.path.join(FLAGS.buckets, FLAGS.train_file))