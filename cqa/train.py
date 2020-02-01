import os

from . import qa
from . import data_utils
from . import utils
from . import evaluate
import tensorflow as tf
from tensorboard import summary as summary_lib

from .qa import PredictMode

_EMBEDDING_PATH = os.path.join(qa.FLAGS.buckets, qa.FLAGS.embedding_path)
_VOCAB_PATH = os.path.join(qa.FLAGS.buckets,  qa.FLAGS.vocab_path)


CONF = tf.ConfigProto()
CONF.gpu_options.allocator_type = 'BFC'

tokenizer = utils.english_tokenizer if qa.FLAGS.use_quora_data else utils.tokenizer
label_mapper = lambda x: int(x) if qa.FLAGS.use_quora_data else data_utils.default_label_mapper(x)


def generate_checkout_point_posfix():
    posfix = []
    if qa.FLAGS.with_margin_loss:
        posfix.append('margin_loss')
    if qa.FLAGS.with_softmax:
        posfix.append('softmax')
    posfix.append('ckpt')
    return '#'.join(posfix)


def train():
    utils.print_argparse_values(qa.FLAGS)
    vocab_id, vocab_info, _ = utils.load_vocab_details(_VOCAB_PATH)

    train_graph = tf.Graph()
    train_session = tf.Session(config=CONF, graph=train_graph)
    with train_graph.as_default(), tf.container("train"):
        embedding = utils._create_pretrained_emb_from_txt(_VOCAB_PATH, _EMBEDDING_PATH)
        model = qa.CQA(vocab_id, embedding, qa.Mode.train)
        model.create_or_load_model(train_session, qa.FLAGS.checkpointDir)

    # dev session for evaluation
    # pr curve summary
    # dev_graph = tf.Graph()
    # dev_session = tf.Session(graph=dev_graph)
    # with dev_graph.as_default(), tf.container("dev"):
    #     predict_result_ph = tf.placeholder(dtype=tf.float32)
    #     ground_ph = tf.placeholder(dtype=tf.bool)
    #
    #     embedding= utils._create_pretrained_emb_from_txt(_VOCAB_PATH, _EMBEDDING_PATH)
    #     dev_model = qa.CQA(vocab_id, embedding, qa.Mode.test)
    #
    #     k = 1
    #     if True:
    #         _, update_op = summary_lib.pr_curve_streaming_op(tag="pr_at_1",
    #                                                          predictions=predict_result_ph,
    #                                                          labels=ground_ph
    #                                                          )
    #         averaged_precision_score_ph = tf.placeholder(dtype=tf.float32, name="averaged_precision_score")
    #         area_under_prcurve_ph = tf.placeholder(dtype=tf.float32, name="area_under_prcurve")
    #
    #         tf.summary.scalar("averaged_precision_score", averaged_precision_score_ph)
    #         tf.summary.scalar("area_under_prcurve", area_under_prcurve_ph)
    #
    #         pr_summary = tf.summary.merge_all()
    #
    #     dev_session.run(tf.local_variables_initializer())

    summary_writer = tf.summary.FileWriter(qa.FLAGS.summaryDir, train_session.graph)

    # read whole training dataset
    dataset = [i for i in data_utils.training_data_generator(os.path.join(qa.FLAGS.buckets, qa.FLAGS.train_file), \
                                                                 qa.FLAGS.batch_size,
                                                                 qa.FLAGS.sentence_len,
                                                                 label_mapper=label_mapper)]

    print('print first 2 training data...')
    for i in dataset[:2]:
        print('length i = ', len(i))
        q1, q2, q3 = i
        print('===============================')
        print('input question: %s' % q1[0])
        print('match question: %s' % q2[1])
        print('match label: %s' % q3[2])
        print('===============================')

    posfix = generate_checkout_point_posfix()
    # start training
    for epoch in range(qa.FLAGS.epoch):
        for q1, q2, labels in dataset[:10]:
            cos_q_a, loss, update_step, summaries = model.train(train_session, q1, q2, q2)
            step = model.global_step.eval(train_session)
            summary_writer.add_summary(summaries, step)
            print("epoch: %s,  step: %s, loss: %s" % (epoch, step, str(loss)))
            if model.global_step.eval(train_session) % 200 == 0:
                output_model_path_prefix = os.path.join(qa.FLAGS.checkpointDir, posfix)
                print("save model to: %s" % ("%s-%s" % (output_model_path_prefix, model.global_step.eval(train_session))))
                model.saver.save(train_session, output_model_path_prefix, global_step=model.global_step)

            # if model.global_step.eval(train_session) % 200 == 0:
            #     if qa.FLAGS.with_softmax:
            #         truncated_predicted_results, truncated_grounded_results, p, r, \
            #         threshold, averaged_precision_score, area_under_prcurve = evaluate.pr_at_k(k, dev_model, dev_session,
            #                                                                                    qa.FLAGS.checkpointDir,
            #                                                                                    os.path.join(qa.FLAGS.buckets, qa.FLAGS.test_file),
            #                                                                                    qa.FLAGS.eval_dir,
            #                                                                                    line_mapper=line_mapper,
            #                                                                                    label_mapper=label_mapper,
            #                                                                                    predictMode=PredictMode.softmax)
            #
            #         if tf_version_1_4_satisfied():
            #             summary_writer.add_summary(dev_session.run(pr_summary,
            #                                                        feed_dict={predict_result_ph: truncated_predicted_results,
            #                                                               ground_ph: [True if int(x) else False for x in truncated_grounded_results],
            #                                                               averaged_precision_score_ph: averaged_precision_score,
            #                                                               area_under_prcurve_ph: area_under_prcurve}),
            #                                        global_step=step)


def do_test():
    utils.print_argparse_values(qa.FLAGS)
    vocab_id, vocab_info, _ = utils.load_vocab_details(_VOCAB_PATH)
    embedding= utils._create_pretrained_emb_from_txt(_VOCAB_PATH, _EMBEDDING_PATH)

    test_session = tf.Session()
    test_model = qa.CQA(vocab_id, embedding, qa.Mode.eval)

    evaluate.pr_at_k(1, test_model, test_session, qa.FLAGS.checkpointDir,
                     os.path.join(qa.FLAGS.buckets, qa.FLAGS.test_file),
                     qa.FLAGS.eval_dir,
                     line_mapper=line_mapper,
                     label_mapper=label_mapper,
                     predictMode=PredictMode.softmax)
    evaluate.pr_at_k(1, test_model, test_session, qa.FLAGS.checkpointDir,
                     os.path.join(qa.FLAGS.buckets, qa.FLAGS.test_file),
                     qa.FLAGS.eval_dir,
                     line_mapper=line_mapper,
                     label_mapper=label_mapper)


def toy_data():
    vocab_id, vocab_info, _ = utils.load_vocab_details("/data/cqa/toy/vocab")
    embedding= utils._create_pretrained_emb_from_txt("/data/cqa/toy/vocab", "/data/cqa/toy/embedding")
    print('vocab: ')
    print(vocab_id)
    print('embedding: ')
    print(embedding)
    model = qa.CQA(vocab_id, embedding, True)
    input_question = [["this", "is", "a"]] * qa.FLAGS.batch_size
    input_answer = [["this", "is", "a", "test"]] * qa.FLAGS.batch_size
    input_wa = [[["what", "is", "a"]] * qa.FLAGS.neg_sample_num] * qa.FLAGS.batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            cos_q_a, cos_q_wa,loss, update_step = model.train(sess, input_question, input_answer, input_wa)
            print(loss)
            print(cos_q_a)
            print(cos_q_wa)


if __name__ == "__main__":
    # toy_data()
    import sklearn
    print("sk_learn version: %s" % sklearn.__version__)
    print("is train: %s" % (not qa.FLAGS.do_test))
    if qa.FLAGS.do_test:
        do_test()
    else:
        train()
