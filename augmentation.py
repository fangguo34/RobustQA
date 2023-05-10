import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering, DistilBertModel
from transformers import AdamW
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

from BackTranslation import BackTranslation
import time
import random
import glob
from os import listdir

def get_dataset_dict(args, datasets, data_dir):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

    return dataset_dict

def combine_dictionaries(args, folder):
    data_final_context = []
    data_final_question = []
    data_final_id = []
    data_final_answer = []

    for file in glob.glob(folder + 'subfile*'):
        print(file)
        tmp = load_json(file)

        data_final_context.append(tmp['context'])
        data_final_question.append(tmp['question'])
        data_final_id.append(tmp['id'])
        data_final_answer.append(tmp['answer'])


    data_final_context_output = [item for sublist in data_final_context for item in sublist]
    data_final_question_output = [item for sublist in data_final_question for item in sublist]
    data_final_id_output = [item for sublist in data_final_id for item in sublist]
    data_final_answer_output = [item for sublist in data_final_answer for item in sublist]

    data_dict = {'context': data_final_context_output, 'question': data_final_question_output,
                'id': data_final_id_output, 'answer': data_final_answer_output}

    return data_dict

def update_dictionary(args, dataset_dict, context_new, question_new):
    context_update = {'context': context_new}
    question_update = {'question': question_new}

    dataset_dict.update(context_update)
    dataset_dict.update(question_update)

    return dataset_dict

def backtranslate_subfile(sub_dict):
    assert len(sub_dict['context']) == len(sub_dict['question'])
    len_instances = len(sub_dict['question'])

    for i in range(len_instances):
        context_trans, question_trans, id_trans, answerstart_trans = backtranslate_instance(sub_dict['context'][i], sub_dict['question'][i], sub_dict['id'][i], sub_dict['answer'][i])

        sub_dict['context'][i] = context_trans
        sub_dict['question'][i] = question_trans
        sub_dict['answer'][i]['answer_start'] = answerstart_trans

    return sub_dict

def backtranslate_instance(context_i, question_i, id_i, answer_i):
    must_keep = answer_i['text']
    must_keep_start = answer_i['answer_start']

    if len(must_keep) == 1:
        # Find the sentence that contains the answer word, change that sentence, get new context paragraph
        word = must_keep[0]
        word_start = must_keep_start[0]
        word_len = len(word)

        before = context_i[0:word_start]
        after = context_i[word_start+word_len:]
        tmp_word = "X" * word_len

        context_new = before + tmp_word + after 

        # Backtransalate sentences before XXX and sentences after XXX (except for the sentence with answer)
        sentence_list = context_new.split('. ')
        idx = [idx for idx, s in enumerate(sentence_list) if tmp_word in s][0]
        sentence_iter_list = ['. '.join(sentence_list[:idx]), sentence_list[idx], '. '.join(sentence_list[idx+1:])]

        sentence_new_list = []
        for sentence in sentence_iter_list:
            if tmp_word in sentence:
                sentence_new = sentence

            else:
                sentence_new = my_backtranslation(sentence)
                # sentence_new = "TEST TEST TEST"

            sentence_new_list.append(sentence_new)
        
        context_trans_tmp = '. '.join(sentence_new_list)    
        id_trans = id_i
        question_trans = my_backtranslation(question_i) # unchanged for now. NEED TO CHANGE LATER TODO
        # question_trans = question_i # unchanged for now. NEED TO CHANGE LATER TODO

        answerstart_trans = [context_trans_tmp.find(tmp_word)]
        print(answerstart_trans)
        context_trans = context_trans_tmp.replace(tmp_word, word)

    elif len(must_keep) > 1:
        # print("oooooops")
    
        context_trans_tmp = []
        id_trans = id_i

        question_trans = my_backtranslation(question_i)
        # question_trans = question_i
        
        answerstart_trans = answer_i['answer_start']
        context_trans = context_i

    # Overall
    # print(context_trans_tmp)
    # print(context_trans)


    return context_trans, question_trans, id_trans, answerstart_trans


def my_backtranslation(input_text):
    trans = BackTranslation(url=['translate.google.com'])
    languages = ['zh-cn', 'zh-tw', 'ja', 'ko', 'fr', 'es', 'pt', 'de', 'ru', 'ar']
    x = 4900

    print(len(input_text))
    input_text_list = [input_text[i: i + x] for i in range(0, len(input_text), x)]
    output = []

    for i in input_text_list:
        print(i) 

        if len(i) > 0 and "www." not in i:
            tmp_lang = random.choice(languages)
            result = trans.translate(i.replace("/", " "), src='en', tmp = tmp_lang)
            output.append(result.result_text)

            time.sleep(2)
        
        else:
            result = i
            output.append(result)

    output_text = '. '.join(output)
    print(output_text)

    return output_text

def write_json(filename, value):
    with open(filename, 'w') as f:
        json.dump(value, f)

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)

    return(data)

def organize_data(args, dataset_dict, save_path):
    s = args.startpos
    n = args.endpos
    # n = len(dataset_dict['question'])
    b = 100

    assert len(dataset_dict['context']) == len(dataset_dict['question'])
    assert s < n

    for i in range(s, n, b):
        context_tmp = {'context': dataset_dict['context'][i:i+b]}
        question_tmp = {'question':  dataset_dict['question'][i:i+b]}
        id_tmp = {'id':  dataset_dict['id'][i:i+b]}
        answer_tmp = {'answer':  dataset_dict['answer'][i:i+b]}

        tmp_dict = {**context_tmp, **question_tmp, **id_tmp, **answer_tmp}
        output_dict = backtranslate_subfile(tmp_dict)
        write_json(save_path + 'subfile_' + str(i), output_dict)    

def main():
    args = get_train_test_args()
    util.set_seed(args.seed)

    if args.task == 'get_dataset_dict':
        dataset_oo_dict = get_dataset_dict(args, 'duorc,race,relation_extraction', 'datasets/oodomain_train')
        write_json('datasets/alldomains_train_dict/oodomain_train_dict', dataset_oo_dict)

        dataset_in_dict = get_dataset_dict(args, 'squad,nat_questions,newsqa', 'datasets/indomain_train')
        write_json('datasets/alldomains_train_dict/indomain_train_dict', dataset_in_dict)

    elif args.task == 'augment_oo': 
        oo_train_dict = load_json('datasets/alldomains_train_dict/oodomain_train_dict')
        oo_train_final_dict = backtranslate_subfile(oo_train_dict)
        write_json('datasets/alldomains_train_dict/oodomain_train_aug_dict', oo_train_final_dict)

    elif args.task == 'augment_in':
        in_train_dict = load_json('datasets/alldomains_train_dict/indomain_train_dict')
        organize_data(args, in_train_dict, 'datasets/alldomains_train_dict/indomain_subfiles/')

    elif args.task == 'combine_augment_in':
        in_train_final_dict = combine_dictionaries(args, 'datasets/alldomains_train_dict/indomain_subfiles/')
        write_json('datasets/alldomains_train_dict/indomain_train_aug_dict', in_train_final_dict)


if __name__ == '__main__':
    main()


# python augmentation.py --startpos 0 --endpos 500 --task 'augment_in'
