import pandas as pd
import numpy as np
import time
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=6) 
model = BertForSequenceClassification(config=config)
character_bert_model = CharacterBertModel.from_pretrained(
    './pretrained-models/medical_character_bert/')
model.bert = character_bert_model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.cuda()
device='cuda'

df_train = pd.read_csv('kannada_offensive.csv')
df_train['sentiment'], uniq = pd.factorize(df_train['sentiment'])
X = df_train['comment'].tolist()
tokenized = [tokenizer.basic_tokenizer.tokenize(text) for text in X]
indexer = CharacterIndexer()  # This converts each token into a list of character indices
input_tensor = indexer.as_padded_tensor(tokenized)
X_train, X_test, y_train, y_test = train_test_split(input_tensor,df_train['sentiment'].tolist(),test_size=0.1,random_state=42)

batch_size = 32

y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
train_data = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_data,batch_size=batch_size)

val_data = TensorDataset(X_test, y_test)
val_dataloader = DataLoader(val_data,batch_size=batch_size)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)

loss_fn = nn.CrossEntropyLoss().to(device)

epochs = 2
train_loss_set = []

for _ in trange(epochs, desc="Epoch"):
  start_time = time.time()
  model.train()
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  for step, batch in enumerate(train_dataloader):

    batch = tuple(t.to(device) for t in batch)
  
    b_input_ids, b_labels = batch

    optimizer.zero_grad()
  
    outputs = model(b_input_ids)[0]
    
    loss = loss_fn(outputs,b_labels)
    train_loss_set.append(loss.item())    
    
    loss.backward()

    optimizer.step()
    
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1
  end_time = time.time()

  print(epoch_time(start_time,end_time))

  print("\nTrain loss: {}".format(tr_loss/nb_tr_steps))

preds = []
with torch.no_grad():
  correct = 0
  total = 0
  for i, batch in enumerate(val_dataloader):
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_labels = batch
    
    outputs = model(b_input_ids)[0]
    # print (outputs)
    prediction = torch.argmax(outputs,dim=1)
    preds.append(prediction)
    total += b_labels.size(0)
    correct+=(prediction==b_labels).sum().item()

final_preds = []
for tensor in preds:
  for pred in tensor:
    final_preds.append(int(pred))

print(classification_report(y_test,final_preds))