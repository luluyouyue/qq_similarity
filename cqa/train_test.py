import os

import qa
import data_utils
import utils
import evaluate
import tensorflow as tf
from tensorboard import summary as summary_lib
import numpy as np

from cqa.qa import PredictMode

'''
_EMBEDDING_PATH = os.path.join(qa.FLAGS.data_dir, "embedding/parenting/vectors_with_no_eos.txt")
_VOCAB_PATH = os.path.join(qa.FLAGS.data_dir,  "embedding/parenting/vocab.txt")
'''

_EMBEDDING_PATH = './data/embedding/parenting/vectors_with_no_eos.txt'
_VOCAB_PATH = './data/embedding/parenting/vocab.txt'
_TRAIN_PATH = './data/cqa/train_q.neg_sampled'
_TEST_PATH = './data/cqa/test_a'

CONF = tf.ConfigProto()  
CONF.gpu_options.allocator_type = 'BFC'

'''
def train():
    print qa.FLAGS.data_dir
    vocab_id, vocab_info, _ = utils.load_vocab_details('./data/embedding/parenting/vocab.txt')
    # f = open('vocab.txt', 'r')
'''

def train():
    vocab_id, vocab_info, _ = utils.load_vocab_details(_VOCAB_PATH)

    train_graph = tf.Graph()
    train_session = tf.Session(config=CONF, graph=train_graph)
    with train_graph.as_default(), tf.container("train"):
        embedding= utils._create_pretrained_emb_from_txt(_VOCAB_PATH, _EMBEDDING_PATH) 
        print type(embedding)
        model = qa.CQA(vocab_id, embedding, qa.Mode.train)
        model.create_or_load_model(train_session, qa.FLAGS.model_dir)

    dev_graph = tf.Graph()
    dev_session = tf.Session(graph=dev_graph)
    with dev_graph.as_default(), tf.container("dev"):
        predict_result_ph = tf.placeholder(dtype=tf.float32)
        ground_ph = tf.placeholder(dtype=tf.bool)

        embedding= utils._create_pretrained_emb_from_txt(_VOCAB_PATH, _EMBEDDING_PATH)
        dev_model = qa.CQA(vocab_id, embedding, qa.Mode.test)

        k = 1
        _, update_op = summary_lib.pr_curve_streaming_op(tag="pr_at_1",
                                                         predictions=predict_result_ph,
                                                         labels=ground_ph
                                                         )
        averaged_precision_score_ph = tf.placeholder(dtype=tf.float32, name="averaged_precision_score")
        area_under_prcurve_ph = tf.placeholder(dtype=tf.float32, name="area_under_prcurve")

        tf.summary.scalar("averaged_precision_score", averaged_precision_score_ph)
        tf.summary.scalar("area_under_prcurve", area_under_prcurve_ph)

        pr_summary = tf.summary.merge_all()

        dev_session.run(tf.local_variables_initializer())

    summary_writer = tf.summary.FileWriter(qa.FLAGS.tb_dir, train_session.graph)
    for epoch in xrange(qa.FLAGS.epoch):
        '''for q, a, wa in data_utils.neg_sampled_data_generator(os.path.join(qa.FLAGS.data_dir, "cqa/train_q.neg_sampled"),
                                                           qa.FLAGS.batch_size,
                                                           qa.FLAGS.neg_sample_num,
                                                           qa.FLAGS.sentence_len):

        '''
        for q, a, wa in data_utils.neg_sampled_data_generator(_TRAIN_PATH,
                                                           qa.FLAGS.batch_size,
                                                           qa.FLAGS.neg_sample_num,
                                                           qa.FLAGS.sentence_len):
            q = np.reshape(np.asarray(map(lambda x: [x] * qa.FLAGS.neg_sample_num, q)), [qa.FLAGS.batch_size * qa.FLAGS.neg_sample_num, qa.FLAGS.sentence_len])
            a = np.reshape(np.asarray(map(lambda x: [x] * qa.FLAGS.neg_sample_num, a)), [qa.FLAGS.batch_size * qa.FLAGS.neg_sample_num, qa.FLAGS.sentence_len])
            wa = np.reshape(np.asarray(wa), [qa.FLAGS.batch_size * qa.FLAGS.neg_sample_num, qa.FLAGS.sentence_len])
            cos_q_a, loss, update_step, summaries = model.train(train_session, q, a, wa)
            step = model.global_step.eval(train_session)
            summary_writer.add_summary(summaries, step)
            print "epoch: %s,  step: %s, loss: %s" % (epoch, step, str(loss))
            if model.global_step.eval(train_session) % 100 == 0:
                print "save model to: %s/ckpt-%s" % (qa.FLAGS.model_dir, model.global_step.eval(train_session))
                model.saver.save(train_session, "%s/ckpt-" % qa.FLAGS.model_dir, global_step=model.global_step)

            if model.global_step.eval(train_session) % 100 == 0:
                # do eval
                if qa.FLAGS.with_margin_loss:
                    truncated_predicted_results, truncated_grounded_results, p, r, threshold, averaged_precision_score, area_under_prcurve = evaluate.pr_at_k(k, dev_model, dev_session, qa.FLAGS.model_dir, evaluate._DEV_Q_PATH, qa.FLAGS.eval_dir)
                    summary_writer.add_summary(dev_session.run(pr_summary,
                                                               feed_dict={predict_result_ph: truncated_predicted_results,
                                                                                      ground_ph: map(lambda x: True if int(x) else False, truncated_grounded_results),
                                                                                      averaged_precision_score_ph: averaged_precision_score,
                                                                                      area_under_prcurve_ph: area_under_prcurve}),
                                               global_step=step)
                if qa.FLAGS.with_softmax:
                    truncated_predicted_results, truncated_grounded_results, p, r, threshold, averaged_precision_score, area_under_prcurve = evaluate.pr_at_k(k, dev_model, dev_session, qa.FLAGS.model_dir, evaluate._DEV_Q_PATH, qa.FLAGS.eval_dir, predictMode=PredictMode.softmax)
                    summary_writer.add_summary(dev_session.run(pr_summary,
                                                               feed_dict={predict_result_ph: truncated_predicted_results,
                                                                          ground_ph: map(lambda x: True if int(x) else False, truncated_grounded_results),
                                                                          averaged_precision_score_ph: averaged_precision_score,
                                                                          area_under_prcurve_ph: area_under_prcurve}),
                                               global_step=step)

def test():
    vocab_id, vocab_info, _ = utils.load_vocab_details(_VOCAB_PATH)
    embedding= utils._create_pretrained_emb_from_txt(_VOCAB_PATH, _EMBEDDING_PATH)

    test_session = tf.Session()
    test_model = qa.CQA(vocab_id, embedding, False)

    evaluate.pr_at_k(1, test_model, test_session, qa.FLAGS.model_dir, evaluate._DEV_Q_PATH, qa.FLAGS.eval_dir)
    evaluate.pr_at_k(1, test_model, test_session, qa.FLAGS.model_dir, evaluate._DEV_Q_PATH, qa.FLAGS.eval_dir, predictMode=PredictMode.softmax)

def toy_data():
    vocab_id, vocab_info, _ = utils.load_vocab_details("/data/cqa/toy/vocab")
    embedding= utils._create_pretrained_emb_from_txt("/data/cqa/toy/vocab", "/data/cqa/toy/embedding")
    print 'vocab: '
    print vocab_id
    print 'embedding: '
    print embedding
    model = qa.CQA(vocab_id, embedding, True)
    input_question = [["this", "is", "a"]] * qa.FLAGS.batch_size
    input_answer = [["this", "is", "a", "test"]] * qa.FLAGS.batch_size
    input_wa = [[["what", "is", "a"]] * qa.FLAGS.neg_sample_num] * qa.FLAGS.batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(200):
            cos_q_a, cos_q_wa,loss, update_step = model.train(sess, input_question, input_answer, input_wa)
            print loss
            print cos_q_a
            print cos_q_wa

if __name__ == "__main__":
    # toy_data()
    print "is train: %s" % (not qa.FLAGS.do_test)
    if qa.FLAGS.do_test:
        test()
    else:  
        train()
if __name__ == "__main__":
    train()

