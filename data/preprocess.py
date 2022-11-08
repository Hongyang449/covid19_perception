import pandas as pd


df_all = pd.read_csv('GCCR002.csv')
#(16087, 132)

# row English only
id_all = df_all['UniqueID']
index1 = ['English' in x for x in id_all]
# col
index2 = ['COVID_diagnosis','Comment_-_changes_in_smell','Comment_-_changes_in_taste','Comment_-_changes_in_chemesthesis','Comment_-_Anything_else_smell__taste__flavor','UniqueID']

df = df_all.iloc[index1,:]
df = df.loc[:,index2]
#(3939, 5)

## class 2/3 positive; class 6 no symptoms; class 5 negative with symptoms
df['COVID_diagnosis'].unique()
#array([4., 2., 6., 5., 1., 3.])

## 2/3 vs 6
index = df['COVID_diagnosis'] == 2 
index += df['COVID_diagnosis'] == 3 
index += df['COVID_diagnosis'] == 6 

df1 = df.loc[index,:]
(df1.iloc[:,0]==6).sum()
#(1528, 5) - 1432 positive & 96 negative

# merge comments
comment = []
for i in range(df1.shape[0]):
    comment.append(str(df1.iloc[i,1]) + '|' + str(df1.iloc[i,2]) + '|' + \
        str(df1.iloc[i,3]) + '|' + str(df1.iloc[i,4]))

df_out = pd.DataFrame({'label':df1['COVID_diagnosis'],
    'comment':comment,
    'id':df1['UniqueID']})

# exclude nan
df_out = df_out.loc[df_out['comment']!='nan|nan|nan|nan',:]

df_out.to_csv('comment_236.tsv', sep='\t', index=False)

(df_out.iloc[:,0]==2).sum()
(df_out.iloc[:,0]==3).sum()
# pos 1085 = 1032 + 53
(df_out.iloc[:,0]==6).sum()
# neg 58
# total = 1143

## 2/3 vs 5
index = df['COVID_diagnosis'] == 2
index += df['COVID_diagnosis'] == 3
index += df['COVID_diagnosis'] == 5

df1 = df.loc[index,:]
(df1.iloc[:,0]==5).sum()
#(1557, 5) - 1432 + 125

# merge comments
comment = []
for i in range(df1.shape[0]):
    comment.append(str(df1.iloc[i,1]) + '|' + str(df1.iloc[i,2]) + '|' + \
        str(df1.iloc[i,3]) + '|' + str(df1.iloc[i,4]))

df_out = pd.DataFrame({'label':df1['COVID_diagnosis'],
    'comment':comment,
    'id':df1['UniqueID']})

# exclude nan
df_out = df_out.loc[df_out['comment']!='nan|nan|nan|nan',:]

df_out.to_csv('comment_235.tsv', sep='\t', index=False)

(df_out.iloc[:,0]==2).sum()
(df_out.iloc[:,0]==3).sum()
# pos 1085 = 1032 + 53
(df_out.iloc[:,0]==5).sum()
# neg 89
# total = 1121


