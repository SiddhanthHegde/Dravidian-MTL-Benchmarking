import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simpletransformers.classification import MultiLabelClassificationModel

df = pd.read_csv('kannada_offensive.csv')
df['sentiment'], uniq = pd.factorize(df['sentiment'])

label = []
for index,row in df.iterrows():
  if row['sentiment'] == 0:
    label.append([1,0,0,0,0,0])
  if row['sentiment'] == 1:
    label.append([0,1,0,0,0,0])
  if row['sentiment'] == 2:
    label.append([0,0,1,0,0,0])
  if row['sentiment'] == 3:
    label.append([0,0,0,1,0,0])
  if row['sentiment'] == 4:
    label.append([0,0,0,0,1,0])
  if row['sentiment'] == 5:
    label.append([0,0,0,0,0,1])

df['labels'] = label

df['text'] = df['comment']
df.drop(['comment'],axis=1,inplace=True)
df.drop('sentiment',axis=1,inplace=True)

train,test = train_test_split(df,test_size=0.1,random_state=42)

model = MultiLabelClassificationModel('xlm', 'xlm-mlm-100-1280',
 num_labels=6, 
 args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 2e-5, 'num_train_epochs': 3, 'max_seq_length': 128})

model.train_model(train)
res, outputs, wrong_preds = model.eval_model(test)

preds = []
for x in outputs:
  preds.append(np.argmax(x))

df_temp = pd.read_csv('kannada_offensive.csv')
df_temp['sentiment'], uniq = pd.factorize(df_temp['sentiment'])
_,val = train_test_split(df_temp,test_size=0.1,random_state=42)

print(classification_report(val['sentiment'].tolist(),preds))


