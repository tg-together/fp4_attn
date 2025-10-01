'''
From https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import os

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def stream_until_tokens(dataset_iter, tokenizer, target_tokens):
    """Stream dataset until we get target_tokens number of tokens."""
    texts = []
    total_tokens = 0
    for sample in dataset_iter:
        texts.append(sample['text'])
        total_tokens += len(tokenizer.encode(sample['text']))
        if total_tokens >= target_tokens:
            break
    return tokenizer("\n\n".join(texts), return_tensors='pt')['input_ids']


def create_dataloader(tokenenc, n_samples, seqlen, batch_size, tokenizer):
    """Add BOS token and create dataloader from tokenized data."""
    tokenenc = tokenenc[0, 1:(n_samples*seqlen)+1].view(n_samples, -1)
    sos_token = tokenizer.bos_token_id
    if sos_token is not None:
        tokenenc = torch.cat((torch.tensor([sos_token]*n_samples).unsqueeze(1), tokenenc), 1)
    dataset = TensorDataset(tokenenc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_wikitext2(nsamples, seed, seqlen, batch_size, model, split='test'):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    if split == 'train':
        # Stream train data for 1000 samples
        traindata = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
        testenc = stream_until_tokens(iter(traindata), tokenizer, seqlen * 500)
        n_samples = min(1000, testenc.shape[1]//seqlen)
    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')['input_ids']
        n_samples = testenc.shape[1]//seqlen
    
    return create_dataloader(testenc, n_samples, seqlen, batch_size, tokenizer)


def get_c4(n_samples, seed, seqlen, batch_size, model, split='validation'):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    if split == 'train':
        # Stream train data for 1000 samples
        traindata = load_dataset('allenai/c4','en',split='train', streaming=True)
        valenc = stream_until_tokens(iter(traindata), tokenizer, seqlen * 1000)
        n_samples = min(1000, valenc.shape[1]//seqlen)
    else:
        if os.path.exists("dumps/c4_valid_joined_ids.pt"):
            valenc = torch.load("dumps/c4_valid_joined_ids.pt")
        else:
            valdata = load_dataset(
                'allenai/c4',
                data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                split='validation')
            valenc = tokenizer("\n\n".join(valdata['text']), return_tensors='pt')['input_ids']
            torch.save(valenc, "dumps/c4_valid_joined_ids.pt")

    
    return create_dataloader(valenc, n_samples, seqlen, batch_size, tokenizer)


def get_pile(n_samples, seed, seqlen, batch_size, model, split='validation'):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    if split == 'train':
        # Stream train data for 1000 samples
        traindata = load_dataset("monology/pile-uncopyrighted", split='train', streaming=True)
        valenc = stream_until_tokens(iter(traindata), tokenizer, seqlen * 1000)
        n_samples = min(1000, valenc.shape[1]//seqlen)
    else:
        if os.path.exists("dumps/pile_test_joined_ids.pt"):
            valenc = torch.load("dumps/pile_test_joined_ids.pt")
        else:
            pile = load_dataset(
                "monology/pile-uncopyrighted",
                data_files={"test": "test.jsonl.zst"},
                split="test",
            )
            pile=pile[:10000]
            valenc = tokenizer("\n\n".join(pile['text']), return_tensors='pt')['input_ids']
            torch.save(valenc, "dumps/pile_test_joined_ids.pt")

    
    return create_dataloader(valenc, n_samples, seqlen, batch_size, tokenizer)


def get_test_tokens(name, nsamples=0, seed=0, seqlen=2048, batch_size=10, model='', train=False):
    split = 'train' if train else 'test'
    
    if name == 'wikitext2':
        return get_wikitext2(nsamples, seed, seqlen, batch_size, model, split=split)
    elif name == 'c4':
        return get_c4(nsamples, seed, seqlen, batch_size, model, split=split)
    elif name == 'pile':
        return get_pile(nsamples, seed, seqlen, batch_size, model, split=split)
