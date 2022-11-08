import os
import sys
import numpy as np
import pandas as pd
import argparse
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModel, utils
from transformers import TextClassificationPipeline
import datasets
import shap
import pickle

# pred and score
def score_record(truth, predictions, input_digits=None):
    if input_digits is None: # bin resolution
        input_digits = 3
    scale=10**input_digits
    pos_values = np.zeros(scale + 1, dtype=np.int64)
    neg_values = np.zeros(scale + 1, dtype=np.int64)
    b = scale+1
    r = (-0.5 / scale, 1.0 + 0.5 / scale)
    all_values = np.histogram(predictions, bins=b, range=r)[0]
    if np.sum(all_values) != len(predictions):
        raise ValueError("invalid values in 'predictions'")
    pred_pos = predictions[truth > 0]
    pos_values = np.histogram(pred_pos, bins=b, range=r)[0]
    pred_neg = predictions[truth == 0]
    neg_values = np.histogram(pred_neg, bins=b, range=r)[0]
    return (pos_values, neg_values)

def calculate_auc(pos_values,neg_values): # auc & auprc; adapted from score2018.py
    tp = np.sum(pos_values)
    fp = np.sum(neg_values)
    tn = fn = 0
    tpr = 1
    tnr = 0
    if tp == 0 or fp == 0:
        # If either class is empty, scores are undefined.
        return (float('nan'), float('nan'))
    ppv = float(tp) / (tp + fp)
    auroc = 0
    auprc = 0
    for (n_pos, n_neg) in zip(pos_values, neg_values):
        tp -= n_pos
        fn += n_pos
        fp -= n_neg
        tn += n_neg
        tpr_prev = tpr
        tnr_prev = tnr
        ppv_prev = ppv
        tpr = float(tp) / (tp + fn)
        tnr = float(tn) / (tn + fp)
        if tp + fp > 0:
            ppv = float(tp) / (tp + fp)
        else:
            ppv = ppv_prev
        auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5
        auprc += (tpr_prev - tpr) * ppv_prev
    return (auroc, auprc)

## ref:
## https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb

## pre-trained model/tokenizer: DistillBERT
## https://arxiv.org/abs/1910.01108
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

path0='../../data/'

df0=pd.read_csv(path0 + 'comment_236_corrected.tsv',sep='\t')
index0= df0.loc[:,'label'] == 6
index1 = df0.loc[:,'label'] != 6
df0.loc[index0,'label']=0
df0.loc[index1,'label']=1

df0.iloc[:,0] = df0.iloc[:,0].astype('int')

# exclude all nan; pre-excluded
index = df0['comment'] != 'nan|nan|nan|nan'
df0 = df0.loc[index,:]

# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-f', '--fold', default=0, type=int, help='cross validtion fold')
    parser.add_argument('-s', '--seed', default=0, type=int, help='seed for train-vali partition')
    args = parser.parse_args()
    return args

args=get_args()

print(sys.argv)
fold = args.fold
seed_partition = args.seed

##############################
## partition
id_all = np.arange(df0.shape[0])
np.random.seed(fold)
np.random.shuffle(id_all)
ratio = [0.8,0.2]
num = int(len(id_all)*ratio[0])
id_tv = id_all[:num]
id_test = id_all[num:]

id_tv.sort()
np.random.seed(seed_partition)
np.random.shuffle(id_tv)
ratio=[0.75,0.25]
num = int(len(id_tv)*ratio[0])
id_train = id_tv[:num]
id_vali = id_tv[num:]

# 0.6 - 0.2 - 0.2
id_train.sort()
id_vali.sort()
id_test.sort()

## prepare dataset
#dict_train={'id':id_train,
#    'label':[df0.iloc[i,0] for i in id_train],
#    'feature':[df0.iloc[i,1] for i in id_train]}
#dict_vali={'id':id_vali,
#    'label':[df0.iloc[i,0] for i in id_vali],
#    'feature':[df0.iloc[i,1] for i in id_vali]}
dict_test={'id':id_test,
    'label':[df0.iloc[i,0] for i in id_test],
    'feature':[df0.iloc[i,1] for i in id_test]}

#dataset_train = datasets.Dataset.from_dict(dict_train)
#dataset_vali = datasets.Dataset.from_dict(dict_vali)
dataset_test = datasets.Dataset.from_dict(dict_test)
# e.g. 
#dataset_vali[0]
#dataset_vali['label'][:10]

##############################
## encoding by pretrained tokenizer    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples['feature'], truncation=True)

#preprocess_function(dataset_train[:5])
#dataset_train_encoded = dataset_train.map(preprocess_function, batched=True)
#dataset_vali_encoded = dataset_vali.map(preprocess_function, batched=True)
dataset_test_encoded = dataset_test.map(preprocess_function, batched=True)

#format othervise torch.tensor() error
#dataset_train_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#dataset_vali_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_test_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(f'model_fold{fold}_seed{seed_partition}', output_attentions=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

## wrap it as a pipeline for shap
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

## pred
#pred_ori = pipe(dataset_test['feature'])
#pred_ori = [x[1]['score'] for x in pred_ori]

## shap
explainer = shap.Explainer(pipe)
shap_values = explainer(dataset_test['feature'])
#shap.plots.text(shap_values[0, :, "POSITIVE"])
pickle.dump(shap_values, open(f'shap_fold{fold}_seed{seed_partition}', 'wb'))
#shap_values=pickle.load(open(f'shap_fold{fold}_seed{seed_partition}', 'rb'))


