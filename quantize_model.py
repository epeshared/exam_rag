import intel_extension_for_pytorch as ipex
import torch
import time
import os
import math
import random
from transformers import AutoTokenizer, AutoModel
from optimum.intel import IPEXModel, INCQuantizer
from datasets import load_dataset, Dataset
from neural_compressor import PostTrainingQuantConfig

# 模型和数据集定义
model_configs = {
    'en': {
        'model_name': 'home/xtang/models/BAAI/bge-reranker-large',
        'quantized_model_path': 'home/xtang/models/BAAI/bge-reranker-large-qtz-int8',
        'dataset_name': 'allenai/qasper'
    },
    'zh': {
        'model_name': 'bge-large-zh-v1.5',
        'quantized_model_path': 'bge-large-zh-v1.5-opt',
        'dataset_name': 'linux-cn/archive'
    }
}

def get_dataset(sample_size, lang='en'):
    """获取指定语言的数据集并随机抽取样本"""
    hfdataset = model_configs[lang]['dataset_name']
    dataset = load_dataset(hfdataset)
    train_set = dataset["train"]
    
    random.seed(666)
    random_samples = random.sample(range(len(train_set)), sample_size)
    
    if lang == 'en':
        random_queries = [random.sample(train_set[x]["qas"]["question"], 1)[0] for x in random_samples]
        random_abstracts = [train_set[x]["abstract"] for x in random_samples]
        samples = random.sample(random_queries + random_abstracts, sample_size)
    else:
        random_queries = [train_set["content"][x] for x in random_samples]
        samples = random.sample(random_queries, sample_size)
    
    random.shuffle(samples)
    
    return Dataset.from_generator(lambda: ({"text": s} for s in samples))

def generate_batch_token(tokenizer, max_length, batch=1, lang='en'):
    """生成批次令牌"""
    dataset = get_dataset(batch, lang)
    sentences = dataset['text']
    tokens = tokenizer(sentences, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    
    return tokens

def quantize(lang='en', sample_size=100):
    """对指定语言的模型进行量化"""
    config = model_configs[lang]
    model = AutoModel.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    calibration_set = get_dataset(sample_size, lang)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

    vectorized_ds = calibration_set.map(preprocess_function, num_proc=10)
    vectorized_ds = vectorized_ds.remove_columns(["text"])
    
    quantizer = INCQuantizer.from_pretrained(model)
    quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", domain="nlp")
    
    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=vectorized_ds,
        save_directory=config['quantized_model_path'],
        batch_size=1,
    )
    
    tokenizer.save_pretrained(config['quantized_model_path'])

# 使用示例：
# 对英文模型进行量化
quantize(lang='en', sample_size=100)

# # 对中文模型进行量化
# quantize(lang='zh', sample_size=100)
