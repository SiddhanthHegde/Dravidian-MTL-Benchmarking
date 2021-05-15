import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
from dataset import MTLdataset
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

def create_data_loader(df,tokenizer,max_len,batch_size,shuffle=True):
  ds = MTLdataset(
    sentence = df['comment'].to_numpy(),
    label1 = df['sent'].to_numpy(),
    label2 = df['off'].to_numpy(),
    tokenizer = tokenizer,
    max_len = max_len
    )

  return DataLoader(ds,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers=4)

def train_epoch(model,data_loader,loss_fn,optimizer,device,n_examples):

  model = model.train()
  losses = []
  correct_predictions = 0

  for data in data_loader:
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    labels = data['label1'].to(device)

    out= model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )

    _, preds1 = torch.max(out, dim=1)

    _, org = torch.max(labels, dim=1)

    loss = loss_fn(out,labels)

    correct_predictions += torch.sum(preds1 == org)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples

def eval_model(model,data_loader, loss, device, n_examples):

  model = model.eval()
  correct_predictions = 0
  f_pred = []
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels = d["label1"].to(device)
      out = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      
      _, preds = torch.max(out, dim=1)
      _, org = torch.max(labels, dim=1)
      loss = loss(out,labels)

      f_pred.append(preds)
        

      correct_predictions += torch.sum(preds == org)


  return correct_predictions.double() / n_examples
	
	
def epoch_time(start_time,end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time/60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins,elapsed_secs

	 
def show_confusion_matrix(confusion_matrix):
	hmap = sns.heatmap(confusion_matrix,annot= True, fmt="d",cmap="Blues")
	hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
	hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
	plt.ylabel('True label')
	plt.xlabel('predicted label')
