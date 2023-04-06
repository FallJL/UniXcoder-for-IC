
# 1.看看run.py怎么搜索代码库，前端传来自然语言字符串content后，输入模型得到向量，与代码库中所有代码的向量进行相似度计算，然后排序返回给前端
# 2.看看代码注释是怎么生成的
# todo 2.实现时，要区分是什么语言，java还是python
# todo 3.实现时，要明确返回多少个代码，根据用户分页的设置来
# todo 4.注意是否存在注入攻击的可能，因为用户的自然语言输入和代码输出都是字符串，需要防范如sql注入的问题


import sys
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from flask import Flask, json, request, jsonify

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 code  # the code presented to the user
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.code = code


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"],
                         js['code'])


def convert_content_to_tensor(content, tokenizer, args):
    """convert search content to token ids"""
    nl = ' '.join(content.split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    nl_ids = torch.tensor(nl_ids)  # convert list to tensor
    nl_ids = nl_ids.unsqueeze(0)  # convert to 2D     (1, 128), not (128)
    return nl_ids


def compute_content_and_code_similarity(content, tokenizer, args, model, code_vecs, code_dataset):
    """compute similarity between search content and all code"""
    nl_input = convert_content_to_tensor(content, tokenizer, args)
    nl_input = nl_input.to(args.device)
    with torch.no_grad():
        nl_vec = model(nl_inputs=nl_input)  # output query vector through the model
        nl_vec = nl_vec.cpu().numpy()

    print("nl_vec.shape:", nl_vec.shape)
    score = np.matmul(nl_vec, code_vecs.T)  # 通过矩阵乘法得到查询对应的每一个代码的相似度得分
    sort_ids = np.argsort(score, axis=-1, kind='quicksort', order=None)[:,::-1]  # 按照相似度降序排序，返回的为索引列表
    print("sort_ids.shape:", sort_ids.shape)

    searched_code = []
    for idx in sort_ids[0, :5]:
        searched_code.append(code_dataset.examples[idx].code)
    return searched_code


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_data():
    print("-------main start------")
    parser = argparse.ArgumentParser()
    lang = "python"

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a json file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=f"dataset/CSN/{lang}/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=f"dataset/CSN/{lang}/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    # model name
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    print("args:", args)

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)

    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.to(args.device)

    # prepare code data
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # Eval!
    logger.info("***** Running search *****")
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()  # pytorch的Module类的eval()方法，设置为评估模式

    '''  # convert all code into vectors at initialization. 
         # if it has already been converted, just load it
         
    code_vecs = []
    count = 1
    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)  # batch[0]是指（代码-描述对）中的代码部分
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            print(f"code_vec.shape {count}:", code_vec.shape)
            count += 1
            code_vecs.append(code_vec.cpu().numpy())  # 得到一组向量，存在列表里
    # model.train()  # pytorch的Module类的train()方法，设置为训练模式
    code_vecs = np.concatenate(code_vecs, 0)
    print("code_vecs.shape:", code_vecs.shape)
    np.save('numpy_CSN_python_codebase', code_vecs)
    '''

    # just load .npy file
    code_vecs = np.load('npy/CSN/python/numpy_CSN_python_codebase.npy')

    # return data
    return {"tokenizer": tokenizer, "args": args, "model": model,
            "code_vecs": code_vecs, "code_dataset": code_dataset}


# @app.route('/codeAI/get-code-search-list', methods=['GET'])
# def get_code_search_list():
#     print('request.args:', request.args)
#     print('content:', request.args.get('content'))
#     content = request.args.get('content')
#     searched_all_code = "class solution{\n" \
#                       "\tprint('hello world')\n" \
#                       "}"
#     return jsonify({'code': searched_all_code})

# start Flask
app = Flask(__name__)
app.data = prepare_data()


@app.route('/codeAI/get-code-search-list', methods=['GET'])
def get_code_search_list():
    print('request.args:', request.args)
    print('content:', request.args.get('content'))
    content = request.args.get('content')

    data = app.data
    searched_code = compute_content_and_code_similarity(content, data["tokenizer"], data["args"], data["model"],
                                                        data["code_vecs"], data["code_dataset"])
    print("searched_code:", searched_code)
    return jsonify({'code': searched_code})


if __name__ == '__main__':
    # warning: Setting debug=True will cause repeated execution, so don't do it
    app.run(port=7777, host='127.0.0.1')
