import os
import sys
import numpy as np
import pandas as pd
import argparse
import pickle

np.set_printoptions(precision=3,suppress=True)

## class 2/3 positive; class 6 no symptoms; class 5 negative with symptoms

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

seed_partition = 0
fold = 0

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
dict_test={'id':id_test,
    'label':[df0.iloc[i,0] for i in id_test],
    'feature':[df0.iloc[i,1] for i in id_test]}
	
## pred
gt_all=np.array([df0.iloc[i,0] for i in id_test])
pred_all=np.load(f'pred/pred_fold{fold}_seed{seed_partition}.npy')

#rank predictions
index=np.argsort(pred_all)
pred_rank = np.zeros(len(index))
pred_rank[index] = np.arange(len(index))
pred_rank = pred_rank / max(pred_rank)

## shap
shap_values=pickle.load(open(f'shap_fold{fold}_seed{seed_partition}', 'rb'))


example_text = []
example_shap = []
example_gt = []
example_pred = []


## 1. positives
np.sum(gt_all==1) 
the_index1 = np.where(gt_all==1)[0]
# sort
tmp=np.argsort(pred_rank[the_index1])[::-1]
the_index1 = the_index1[tmp]
pred_rank[the_index1]
text_pos = df0.iloc[id_test[the_index1],1].tolist()
for i in range(len(the_index1)):
    print(i, pred_rank[the_index1][i], text_pos[i])

i = 6
the_text = text_pos[i].replace('|',' ').replace('nan','').rstrip()
len(the_text.split(' '))
the_word = shap_values.data[the_index1[i]]
the_index = ~np.array([x=='' or x=='|' or x=='nan' or x=='. ' or x=='’' or x=='t ' or x=='m ' \
    or x=='vn' or x=="'" or x=='(' or x==')' or x==', ' or x=='.' or x=='s ' or x=='d ' or x=='ed' for x in the_word])
np.array(shap_values.data[the_index1[i]])[the_index]
the_shap = shap_values.values[the_index1[i]][:,1][the_index]
the_shap.shape
example_text.append(the_text)
example_shap.append(' '.join(np.round(the_shap, decimals=3).astype('str').tolist()))
example_gt.append(1)
example_pred.append(np.round(pred_rank[the_index1][i], decimals=3))

i=24
the_text = text_pos[i].replace('|',' ').replace('nan','').replace('!',' !').rstrip()
the_word = shap_values.data[the_index1[i]]
the_index = ~np.array([x=='' or x=='|' or x=='nan' or x=='. ' or x=='’' or x=='t ' or x=='m ' \
    or x=='vn' or x=="'" or x=='(' or x==')' or x==', ' or x=='.' or x=='ers' for x in the_word])
the_shap = shap_values.values[the_index1[i]][:,1][the_index]
# check length
len(the_text.split(' '))
the_shap.shape
# check values
the_text.split(' ')
np.array(shap_values.data[the_index1[i]])[the_index]
the_shap
example_text.append(the_text)
example_shap.append(' '.join(np.round(the_shap, decimals=3).astype('str').tolist()))
example_gt.append(1)
example_pred.append(np.round(pred_rank[the_index1][i], decimals=3))




## 2. negatives
np.sum(gt_all==0) #11
the_index0 = np.where(gt_all==0)[0]
# sort
tmp=np.argsort(pred_rank[the_index0])
the_index0 = the_index0[tmp]
#array([142,  14, 195, 211, 217, 151, 115,  71, 105,  50, 178])
pred_rank[the_index0]
#array([0.018, 0.022, 0.035, 0.088, 0.118, 0.171, 0.272, 0.364, 0.377, 0.39 , 0.522])
text_neg = df0.iloc[id_test[the_index0],1].tolist()
for i in range(len(the_index0)):
    print(i, pred_rank[the_index0][i], text_neg[i])
    shap_values.data[the_index0[i]]
    shap_values.values[the_index0[i]][:,1]



i=0
the_text = text_neg[i].replace('|',' ').replace('nan','').replace('!',' !').rstrip().lstrip()
the_word = shap_values.data[the_index0[i]]
the_index = ~np.array([x=='' or x=='|' or x=='nan' or x=='. ' or x=='’' or x=='t ' or x=='m ' \
    or x=='vn' or x=="'" or x=='(' or x==')' or x==', ' or x=='.' or x=='ers' or x=='ve ' \
    or x=='gging ' for x in the_word])
the_shap = shap_values.values[the_index0[i]][:,1][the_index]
# check length
len(the_text.split(' '))
the_shap.shape
# check values
the_text.split(' ')
np.array(shap_values.data[the_index0[i]])[the_index]
the_shap
example_text.append(the_text)
example_shap.append(' '.join(np.round(the_shap, decimals=3).astype('str').tolist()))
example_gt.append(0)
example_pred.append(np.round(pred_rank[the_index0][i], decimals=3))


i=2
the_text = text_neg[i].replace('|',' ').replace('nan','').replace('!',' !').replace('none ','').rstrip().lstrip()
the_word = shap_values.data[the_index0[i]]
the_index = ~np.array([x=='' or x=='|' or x=='nan' or x=='. ' or x=='’' or x=='t ' or x=='m ' \
    or x=='vn' or x=="'" or x=='(' or x==')' or x==', ' or x=='.' or x=='ers' or x=='ve ' \
    or x=='gging ' or x=='none' for x in the_word])
the_shap = shap_values.values[the_index0[i]][:,1][the_index]
# check length
len(the_text.split(' '))
the_shap.shape
# check values
the_text.split(' ')
np.array(shap_values.data[the_index0[i]])[the_index]
the_shap
example_text.append(the_text[:-3])
example_shap.append(' '.join(np.round(the_shap[:-1], decimals=3).astype('str').tolist()))
example_gt.append(0)
example_pred.append(np.round(pred_rank[the_index0][i], decimals=3))


i=9
the_text = text_neg[i].replace('|',' ').replace('nan','').replace('!',' !').rstrip().lstrip()
the_word = shap_values.data[the_index0[i]]
the_index = ~np.array([x=='' or x=='|' or x=='nan' or x=='. ' or x=='’' or x=='t ' or x=='m ' \
    or x=='vn' or x=="'" or x=='(' or x==')' or x==', ' or x=='.' or x=='ers' or x=='ve ' \
    or x=='gging ' or x=='none' or x=='1' or x=='/' for x in the_word])
the_shap = shap_values.values[the_index0[i]][:,1][the_index]
# check length
len(the_text.split(' '))
the_shap.shape
# check values
the_text.split(' ')
np.array(shap_values.data[the_index0[i]])[the_index]
the_shap
example_text.append(the_text)
example_shap.append(' '.join(np.round(the_shap, decimals=3).astype('str').tolist()))
example_gt.append(0)
example_pred.append(np.round(pred_rank[the_index0][i], decimals=3))


i=10
the_text = text_neg[i].replace('|',' ').replace('nan','').replace('!',' !').replace('  ',' ').replace('a n','an').rstrip().lstrip()
the_word = shap_values.data[the_index0[i]]
the_index = ~np.array([x=='' or x=='|' or x=='nan' or x=='. ' or x=='’' or x=='t ' or x=='m ' \
    or x=='vn' or x=="'" or x=='(' or x==')' or x==', ' or x=='.' or x=='ers' or x=='ve ' \
    or x=='gging ' or x=='n ' or x=='chy ' or x=='cus ' or x=='.  ' for x in the_word])
the_shap = shap_values.values[the_index0[i]][:,1][the_index]
# check length
len(the_text.split(' '))
the_shap.shape
# check values
the_text.split(' ')
np.array(shap_values.data[the_index0[i]])[the_index]
the_shap
example_text.append(the_text)
example_shap.append(' '.join(np.round(the_shap, decimals=3).astype('str').tolist()))
example_gt.append(0)
example_pred.append(np.round(pred_rank[the_index0][i], decimals=3))

df = pd.DataFrame({'gt':example_gt, 'pred':example_pred, 'text':example_text, 'shap':example_shap})
df.to_csv('example.tsv', sep='\t', index=False)


