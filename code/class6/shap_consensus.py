import os
import sys
import numpy as np
import pandas as pd
import pickle

np.set_printoptions(precision=3,suppress=True)

## class 2/3 positive; class 6 no symptoms; class 5 negative with symptoms
## 1432 positive & 96 negative
## negative shap values contribute to positive case prediction

# raw shap values
seed_partition=0
shap_all = np.zeros(0)
word_all = []
for i in range(10):
	print(i)
    shap_values=pickle.load(open(f'shap_fold{i}_seed{seed_partition}', 'rb'))
	for j in range(len(shap_values)):
		shap_all = np.concatenate((shap_all,shap_values[j].values[:,0]))
		word_all += shap_values[j].data

# remove extra spaces
word_all = [x.replace(' ','') for x in word_all]
# to lower cases
word_all = [x.lower() for x in word_all]
# upper I
word_all[word_all=='i']='I'

word_all = np.array(word_all)
word_all.shape #(167183,)
word_unique,num_count = np.unique(word_all,return_counts=True)
word_unique.shape #(3367,)

raw_all=[]
avg_all=[]
for i in range(word_unique.shape[0]):
	the_word = word_unique[i]
	the_num = num_count[i]
	the_index = word_all == the_word
	the_raw = shap_all[the_index]
	the_avg = the_raw.sum() / the_num
	# save
	the_concat = '_'.join(['%.2f' % x for x in the_raw.tolist()])
	raw_all.append(the_concat)
	avg_all.append(the_avg)
	print(the_word, the_num, the_avg)

df = pd.DataFrame({'word':word_unique.tolist(),'count':num_count.tolist(),'shap_avg':avg_all,'shap_raw':raw_all})
df.to_csv('shap_consensus.tsv', sep='\t', index=False) 

## total number of examples = 2290 = 229 * 10fold
## top words that have a frequency >10%
the_index = num_count >=229
df_top = df.loc[the_index,:]
## exclude symbols and numbers..
word_unique[the_index]
df_top = df_top.iloc[10:-3,:]

df_top.to_csv('shap_consensus_top.tsv', sep='\t', index=False) 

df_top_abs = df_top.iloc[:,:3]
df_top_abs['shap_avg'] = df_top_abs['shap_avg'].abs()
df_top_abs.to_csv('tbl_shap_class6.csv', index=False)



