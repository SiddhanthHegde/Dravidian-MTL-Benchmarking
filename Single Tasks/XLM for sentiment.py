import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simpletransformers.classification import MultiLabelClassificationModel

df = pd.read_csv('kannada_sentiment.csv')
df['sentiment'], uniq = pd.factorize(df['sentiment'])
df['Positive'] = 0
df['not-Kannada'] = 0
df['unknown'] = 0
df['Negative'] = 0
df['Mixed'] = 0

for index,row in df.iterrows():
  if row['sentiment'] == 0:
    df['Positive'].iloc[index] = 1
  if row['sentiment'] == 1:
    df['not-Kannada'].iloc[index] = 1
  if row['sentiment'] == 2:
    df['unknown'].iloc[index] = 1
  if row['sentiment'] == 3:
    df['Negative'].iloc[index] = 1
  if row['sentiment'] == 4:
    df['Mixed'].iloc[index] = 1

df.drop('sentiment',axis=1,inplace=True)
df['labels'] = list(zip(df['Positive'].tolist(), df['not-Kannada'].tolist(), df['unknown'].tolist(), df['Negative'].tolist(),  df['Mixed'].tolist()))
df['text'] = df['comment']
df.drop(['comment','Positive','not-Kannada','unknown','Negative','Mixed'],axis=1,inplace=True)

train,test = train_test_split(df,test_size=0.1,random_state=42)


model = MultiLabelClassificationModel('xlm', 'xlm-mlm-100-1280',
 num_labels=5, 
 args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 2e-5, 'num_train_epochs': 3, 'max_seq_length': 128})

model.train_model(train)
res, outputs, wrong_preds = model.eval_model(test)

preds = []
for x in outputs:
  preds.append(np.argmax(x))

df_temp = pd.read_csv('kannada_sentiment.csv')
df_temp['sentiment'], uniq = pd.factorize(df_temp['sentiment'])
_,val = train_test_split(df_temp,test_size=0.1,random_state=42)

print(classification_report(val['sentiment'].tolist(),preds))

