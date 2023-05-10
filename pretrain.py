import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer
from transformers import DistilBertForMaskedLM
from transformers import AdamW
from tensorboardX import SummaryWriter


from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

import random


class PreTrainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def train(self, model, train_dataloader):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels_ids = batch['labels_ids'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels_ids)
                    loss = outputs.loss
                    loss.backward()
                    optim.step()
                    
                    self.save(model)

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)

                    global_idx += 1

def get_dataset(args, datasets, data_dir, tokenizer):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)


    contexts_list = list(map(split_context, set(dataset_dict['context'])))
    contexts = [item for sublist in contexts_list for item in sublist]
    questions = dataset_dict['question']    
    pretrain_labels = questions
    # print(pretrain_labels)

    # randomly mask inputs
    pretrain_inputs = list(map(maskdata, pretrain_labels))

    # tokenize inputs 
    inputs_tokens = tokenizer(pretrain_inputs, truncation=True, max_length=128, padding='max_length', return_tensors="pt")
    inputs_tokens['labels_ids'] = tokenizer(pretrain_labels, truncation=True, max_length=128, padding='max_length', return_tensors="pt")['input_ids']

    
    return util.MLMDataset(inputs_tokens)

def split_context(context):
    context_new = context.split('.')

    return context_new

def maskdata(sentence):
    words = open('wordlist.10000.txt').read().split('\n')

    sentence_split = sentence.split(' ')
    sentence_len = len(sentence_split)
    sentence_len_15 = round(sentence_len*0.15)

    mask_ids = random.sample(range(sentence_len), sentence_len_15)

    for i in mask_ids:
        if random.uniform(0,1) < 0.8:
            sentence_split[i] = '[MASK]'

        elif random.uniform(0,1) >= 0.8 and random.uniform(0,1) < 0.9:
            sentence_split[i] = random.choice(words)

    sentence_new = ' '.join(sentence_split)

    return sentence_new


def main():
    # Get arguments   
    args = get_train_test_args()
    util.set_seed(args.seed)

    # Get a pre-trained model
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # logistics 
    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda')

    # get additional pretraining data
    train_dataset = get_dataset(args, args.train_datasets, args.train_dir, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset))

    # continued pretraining
    pretrainer = PreTrainer(args, log)
    pretrainer.train(model, train_loader)

if __name__ == '__main__':
    main()


# python pretrain.py --do-train --train-datasets duorc,race,relation_extraction --run-name continued_pretrain --train-dir 'datasets/oodomain_train' 
