# -*- coding: utf-8 -*-
import sys
import argparse
import math

def data_analysis(file_path):
    f = open(file_path,'r')
    lines = f.readlines()
    print len(lines)
    # 
    dict_sentence = {}
    dict_label = {}
    for line in lines:
        items = line.strip().split('\t')
        q1 = items[0].split(' ')
        q2 = items[1].split(' ')
        label = int(items[2])
        
        # 统计句子单词分布
        q1_len = int(math.ceil(float(len(q1))/2))
        q2_len = int(math.ceil(float(len(q2))/2))
        if q1_len in dict_sentence.keys():
            dict_sentence[q1_len] += 1
            # print 'the key exits', q1_len, dict_sentence[q1_len]
        else:
            dict_sentence[q1_len] = 1
        if q2_len in dict_sentence.keys():
            dict_sentence[q2_len] += 1
            # print 'the key exits', q2_len, dict_sentence[q2_len]
        else:
            dict_sentence[q2_len] = 1

        # print 'the key not exits', q1_len, dict_sentence[q1_len]
        
        # 统计正负样本分布
        if label in dict_label.keys():
            dict_label[label] += 1
        else:
            dict_label[label] = 1
    return dict_sentence, dict_label
    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='train', help='the input file path')
    args = parser.parse_args()
    
    file_path = args.file
    
    dict_sentence, dict_label = data_analysis(file_path)
    
    for sentence_len, sentence_len_num in dict_sentence.items():
        print sentence_len, ':', sentence_len_num
    for label, label_num in dict_label.items():
        print label, ':', label_num
if __name__=='__main__':
    main(sys.argv)
