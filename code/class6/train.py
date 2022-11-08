import os
import sys
import numpy as np
import pandas as pd
import argparse
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import datasets

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
dict_train={'id':id_train,
    'label':[df0.iloc[i,0] for i in id_train],
    'feature':[df0.iloc[i,1] for i in id_train]}
dict_vali={'id':id_vali,
    'label':[df0.iloc[i,0] for i in id_vali],
    'feature':[df0.iloc[i,1] for i in id_vali]}
dict_test={'id':id_test,
    'label':[df0.iloc[i,0] for i in id_test],
    'feature':[df0.iloc[i,1] for i in id_test]}

# oversample
train_pos = id_train[df0.iloc[id_train,0] == 1]
train_neg = id_train[df0.iloc[id_train,0] == 0]

if len(train_neg) > len(train_pos):
    num_diff = len(train_neg) - len(train_pos)
    index = np.random.randint(0, len(train_pos), num_diff)
    id_train_all = np.concatenate((train_neg, train_pos, train_pos[index]))
else:
    num_diff = len(train_pos) - len(train_neg)
    index = np.random.randint(0, len(train_neg), num_diff)
    id_train_all = np.concatenate((train_neg, train_pos, train_neg[index]))

dict_train={'id':id_train_all,
    'label':[df0.iloc[i,0] for i in id_train_all],
    'feature':[df0.iloc[i,1] for i in id_train_all]}

dataset_train = datasets.Dataset.from_dict(dict_train)
dataset_vali = datasets.Dataset.from_dict(dict_vali)
dataset_test = datasets.Dataset.from_dict(dict_test)
# e.g. 
#dataset_vali[0]
#dataset_vali['label'][:10]

# shuffle
dataset_train=dataset_train.shuffle()
dataset_vali=dataset_vali.shuffle()
##############################


##############################
## encoding by pretrained tokenizer    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples['feature'], truncation=True)

#preprocess_function(dataset_train[:5])
dataset_train_encoded = dataset_train.map(preprocess_function, batched=True)
dataset_vali_encoded = dataset_vali.map(preprocess_function, batched=True)
dataset_test_encoded = dataset_test.map(preprocess_function, batched=True)

#format othervise torch.tensor() error
dataset_train_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_vali_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_test_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## fine tune
#class MultilabelTrainer(Trainer):
#    def compute_loss(self, model, inputs, return_outputs=False):
#        labels = inputs.pop("labels")
#        outputs = model(**inputs)
#        logits = outputs.logits
#        loss_fct = torch.nn.BCEWithLogitsLoss()
#        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
#                        labels.float().view(-1, self.model.config.num_labels))
#        return (loss, outputs) if return_outputs else loss

num_class = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_class)

args = TrainingArguments(
    "fine_tune",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    #learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #num_train_epochs=10,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric=datasets.load_metric('pearsonr')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) # for >= 2 classes
    #predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train_encoded,
    eval_dataset=dataset_vali_encoded,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()
#trainer.evaluate()

## pred and score
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

pred_ori=[]
for i in range(dataset_test.shape[0]):
    inputs = tokenizer(dataset_test[i]['feature'],truncation=True,return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    #pred_ori.append(outputs.logits.item())
    #pred_ori.append(outputs.logits.tolist()[0][0])
    pred_ori.append(torch.softmax(outputs.logits, dim=1).tolist()[0][1])

pred_ori=np.array(pred_ori)
pred_all=(pred_ori - np.min(pred_ori)) / (np.max(pred_ori) - np.min(pred_ori))
gt_all=np.array(dataset_test['label'])

#pred_all = 1-pred_all
reso_digits=5 # auc resolution
positives, negatives = score_record(gt_all.flatten(),pred_all.flatten(),reso_digits)
auc, auprc = calculate_auc(positives, negatives)
pear = np.corrcoef(pred_all, gt_all)[0,1]
auc,auprc
pear


print('fold%d: auroc=%.3f, auprc=%.3f, pear=%.3f' % (fold,auc,auprc,pear))

os.system('mkdir -p pred')
np.save(f'pred/pred_fold{fold}_seed{seed_partition}', pred_all)

## save and load 
model.save_pretrained(f'model_fold{fold}_seed{seed_partition}')

#model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_class)
#model = AutoModelForSequenceClassification.from_pretrained(f'model_fold{fold}_seed{seed_partition}')
#model.to(device)

#rank predictions
index=np.argsort(pred_all)
pred_rank = np.zeros(len(index))
pred_rank[index] = np.arange(len(index))
pred_rank = pred_rank / max(pred_rank)

positives, negatives = score_record(gt_all.flatten(),pred_rank.flatten(),reso_digits)
auc, auprc = calculate_auc(positives, negatives)
pear = np.corrcoef(pred_rank, gt_all)[0,1]
auc,auprc
pear

print('after reseting values based on ranking')
print('fold%d: auroc=%.3f, auprc=%.3f, pear=%.3f' % (fold,auc,auprc,pear))





