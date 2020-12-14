import re
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

def preprocess_texts(sentences):
  sentences = [re.sub(r'http\S+','',s) for s in sentences]
  sentences = [s.replace('#','') for s in sentences]
  sentences = [s + " [SEP] [CLS]" for s in sentences]
  return sentences

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

df_train = pd.read_csv('kannada_offensive.csv')
df_train['sentiment'], uniq = pd.factorize(df_train['sentiment'])

X = df_train['comment'].values.tolist()
y = df_train['sentiment'].values.tolist()
X = preprocess_texts(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

tokenized_texts_train = [tokenizer.tokenize(s) for s in X_train]
tokenized_texts_val = [tokenizer.tokenize(s) for s in X_test]
print (tokenized_texts_train[0])

MAX_LEN = 128
input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_train]
input_ids_val = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_val]

input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_val = pad_sequences(input_ids_val, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks_train = []
attention_masks_val = []

for seq in input_ids_train:
  seq_mask = [float(i>0) for i in seq]
  attention_masks_train.append(seq_mask)

for seq in input_ids_val:
  seq_mask = [float(i>0) for i in seq]
  attention_masks_val.append(seq_mask)

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=5)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)

train_loss_set = []

epochs = 5

for _ in trange(epochs, desc="Epoch"):
  start_time = time.time()
  model.train()
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  for step, batch in enumerate(train_dataloader):

    batch = tuple(t.to(device) for t in batch)
  
    b_input_ids, b_input_mask, b_labels = batch

    optimizer.zero_grad()
  
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    
    loss = outputs[0]
    logits = outputs[1]
    train_loss_set.append(loss.item())    
    
    loss.backward()

    optimizer.step()
    
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1
  end_time = time.time()

  print(epoch_time(start_time,end_time))

  print("\nTrain loss: {}".format(tr_loss/nb_tr_steps))
  #print("\nTrain accuracy : {}".format(100 * correct / total))

preds = []
with torch.no_grad():
  correct = 0
  total = 0
  for i, batch in enumerate(validation_dataloader):
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch
    
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    # print (outputs)
    prediction = torch.argmax(outputs[0],dim=1)
    preds.append(prediction)
    total += b_labels.size(0)
    correct+=(prediction==b_labels).sum().item()

final_preds = []
for tensor in preds:
  for pred in tensor:
    final_preds.append(int(pred))

print(classification_report(y_test,final_preds))








