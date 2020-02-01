# encoding=utf8
import codecs
import os
import re

import tensorflow as tf
import random

from . import utils

'''
所有qa数据在 线上mysql znyd qa_answer, qa_question, qa_rule， 通过rule id 关联。

qa_rule为input question, qa_answer 为 rule的answer， qa_question 为相关问题（相似问题），目前应该只有rule 本身

所有标注数据在 bmcsc sample
'''

def _clean_unistr(unistr):
    return unistr.replace("\t", " ").replace("\n", " ").encode("utf8")

def pad_sent_se(sent, sent_len):
    res = [utils.SOS]
    res += sent
    res.append(utils.EOS)
    if len(res) > sent_len:
        res = res[:sent_len]
    else:
        res += [utils.PAD] * (sent_len-len(res))
    assert len(res) == sent_len
    return res

def generate_formatted_data(tokenized_raw_data_path, output_path):
    """
    generate train test and dev data
    :return:
    """
    trainQf = open(os.path.join(output_path, "train_q"), "w")
    trainAf = open(os.path.join(output_path, "train_a"), "w")

    devQf = open(os.path.join(output_path, "dev_q"), "w")
    devAf = open(os.path.join(output_path, "dev_a"), "w")

    testQf = open(os.path.join(output_path, "test_q"), "w")
    testAf = open(os.path.join(output_path, "test_a"), "w")

    dataset = []
    # group data by input question
    current_question = None
    cur_datum = []
    for line in open(tokenized_raw_data_path):
        # id	question	answer	answerContent	rankId
        linfo = line.split("\t")[1:]
        if "rankId" in line:
            continue
        if int(linfo[-1]) not in [1, 2, 3]:
            continue
        if linfo[0] == current_question:
            cur_datum.append(linfo)
        else:
            dataset.append(cur_datum)
            cur_datum = []
            current_question = linfo[0]

    dataset.append(cur_datum)
    random.shuffle(dataset)
    data_size = len(dataset)

    for dat in dataset[:int(data_size * 0.8)]:
        for d in dat:
            trainQf.write("%s\t%s\t%s\n" % (d[0], d[1], d[-1].strip()))
            trainAf.write("%s\t%s\t%s\n" % (d[0], d[2], d[-1].strip()))

    for dat in dataset[int(data_size * 0.8) : int(data_size * 0.9)]:
        for d in dat:
            devQf.write("%s\t%s\t%s\n" % (d[0], d[1], d[-1].strip()))
            devAf.write("%s\t%s\t%s\n" % (d[0], d[2], d[-1].strip()))

    for dat in dataset[int(data_size * 0.9):]:
        for d in dat:
            testQf.write("%s\t%s\t%s\n" % (d[0], d[1], d[-1].strip()))
            testAf.write("%s\t%s\t%s\n" % (d[0], d[2], d[-1].strip()))

    trainAf.close()
    trainQf.close()
    devAf.close()
    devQf.close()
    testAf.close()
    testQf.close()


def neg_sample_training_data(training_data_path, neg_sample_num, label_mapper):
    oof = open("%s.neg_sampled" % training_data_path, 'w')

    # get all questions
    total_question = set()
    with open(training_data_path) as iif:
        for line in iif:
            line_info = line.split("\t")
            if len(line_info) != 3:
                print("illegal line: " + line)
                continue
            total_question.add(line_info[0].strip())
            total_question.add(line_info[1].strip())

    total_question = list(total_question)
    # do neg sample if neg counterparts are less than FLAGS.neg_sample_num
    current_question = None
    cur_pos_answer = set()
    cur_neg_answer = set()
    cur_neg_sample_id = 0
    sampled_dataset=[]
    neg_sample_info = {}
    for line in open(training_data_path):
        # question	answer	rankId
        linfo = line.split("\t")
        if linfo[0] == current_question:
            # pos example
            if label_mapper(linfo[-1]) == 1:
                cur_pos_answer.add(linfo[1])
            else:
                cur_neg_answer.add(linfo[1])
        else:
            # do neg sampling
            if current_question is not None:
                if neg_sample_num < len(cur_neg_answer):
                    cur_neg_answer = list(cur_neg_answer)[:]
                else:
                    while neg_sample_num > len(cur_neg_answer):
                        random.shuffle(total_question)
                        for rd in total_question[:neg_sample_num - len(cur_neg_answer)]:
                            cur_neg_answer.add(rd)

                for a in cur_pos_answer:
                    sampled_dataset.append((a, current_question, cur_neg_sample_id))
                    sampled_dataset.append((current_question, a, cur_neg_sample_id))

                neg_sample_info[cur_neg_sample_id] = cur_neg_answer
            current_question = linfo[0]
            cur_neg_answer = set()
            cur_pos_answer = set()
            cur_neg_sample_id += 1

    random.shuffle(sampled_dataset)

    for d in sampled_dataset:
        # pos sample
        oof.write("%s\t%s\t1\n" % (d[0], d[1]))
        # neg sample
        neg_datum = neg_sample_info[d[-1]]
        assert len(neg_datum) == neg_sample_num, 'get %s, should be %s' % (len(neg_datum), neg_sample_num)
        for n in neg_datum:
            oof.write("%s\t%s\t0\n" % (d[0], n))
    oof.close()


tokenizer = utils.tokenizer


def neg_sampled_data_generator(neg_sampled_data_path, batch_size, neg_sample_num, sentence_length):
    batched_input_question = []
    batched_input_answer = []
    batched_input_wrong_answer = []
    input_wrong_answer = []
    with (codecs.getreader("utf-8"))(tf.gfile.Open(neg_sampled_data_path, "rb")) as iif:
        for line in iif:
            line = line.strip()
            if len(line) == 0:
                continue
            frags = line.split("\t")
            assert len(frags) == 3

            if int(frags[-1]) == 0:
                input_wrong_answer.append(pad_sent_se(tokenizer(frags[1]), sentence_length))
            else:
                if len(input_wrong_answer) > 0:
                    if len(input_wrong_answer) < neg_sample_num:
                        print("insufficient data for neg sample.")
                    else:
                        batched_input_wrong_answer.append(input_wrong_answer[:neg_sample_num])
                        input_wrong_answer = []

                if len(batched_input_question) == batch_size:
                    assert len(batched_input_question) == len(batched_input_answer) == len(batched_input_wrong_answer) == batch_size
                    yield batched_input_question, batched_input_answer, batched_input_wrong_answer
                    batched_input_question = []
                    batched_input_answer = []
                    batched_input_wrong_answer = []

                batched_input_answer.append(pad_sent_se(tokenizer(frags[1]), sentence_length))
                batched_input_question.append(pad_sent_se(tokenizer(frags[0]), sentence_length))


def training_data_generator(training_data_path, batch_size, sentence_length, label_mapper=None):
    batched_input_question = []
    batched_input_answer = []
    batch_labels = []
    with (codecs.getreader("utf-8"))(tf.gfile.Open(training_data_path, "rb")) as iif:
        for line in iif:
            line = line.strip()
            if len(line) == 0:
                continue
            frags = line.split("\t")
            assert len(frags) == 3

            if len(batched_input_question) == batch_size:
                assert len(batched_input_question) == len(batched_input_answer) == len(batch_labels) == batch_size
                yield batched_input_question, batched_input_answer, batch_labels
                batched_input_question = []
                batched_input_answer = []
                batch_labels = []

            batched_input_answer.append(pad_sent_se(tokenizer(frags[1]), sentence_length))
            batched_input_question.append(pad_sent_se(tokenizer(frags[0]), sentence_length))
            # from 1, 2, 3 to [0, 1, 2]
            if not label_mapper:
                batch_labels.append(int(frags[-1]) - 1)
            else:
                batch_labels.append(label_mapper(frags[-1]))


def extract_vocab_from_word2vec(path_to_word2vec_path, vocab_path):
    oof = open(vocab_path, 'w')
    oof.write("%s\n" % utils.UNK)
    oof.write("%s\n" % utils.SOS)
    oof.write("%s\n" % utils.EOS)
    oof.write("%s\n" % utils.PAD)

    with open(path_to_word2vec_path) as iif:
        for line in iif:
            frags = re.split(r"\s+", line.strip())
            if len(frags) == 2:
                continue
            else:
                oof.write("%s\n" % frags[0])

    oof.close()


def default_label_mapper(x):
    if 1 == x:
        return 0
    else:
        return 1


if __name__ == "__main__":
    # dump_raw_qqa_data_from_mysql("/data/cqa/raw/raw_data.txt")
    # generate_formatted_data("/data/cqa/raw/raw_data.txt.tokenized", "/data/cqa/qa_separated")
    # neg_sample_training_data("/data/cqa/qa_separated/train_q", cqa.FLAGS.neg_sample_num, lambda x: 1 if int(x) == 3 else 0)

    # extract_vocab_from_word2vec("/data/embedding/parenting/vectors_with_no_eos.txt", "/data/embedding/parenting/vocab.txt")
    count = 0
    for q, a, l in training_data_generator("/Users/zechuan/zhidian/zd-parent/py/data/cqa/train_q", 2, 20):
        print("batch=============")
        print('   '.join([' '.join(x) for x in q]))
        print('   '.join([' '.join(x) for x in a]))
        print(l)
        print()

        count += 1
        if count == 3:
            break