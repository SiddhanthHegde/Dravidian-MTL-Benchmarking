import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
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

def train_epoch(model,data_loader,loss_1,loss_2,optimizer,device,scheduler,n_examples):
  model = model.train()
  losses = []
  correct_predictions1 = 0
  correct_predictions2 = 0

  for data in data_loader:
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    labels1 = data['label1'].to(device)
    labels2 = data['label2'].to(device)

    out1,out2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
    _, preds1 = torch.max(out1, dim=1)
    _, preds2 = torch.max(out2, dim=1)

    _, org1 = torch.max(labels1, dim=1)
    _, org2 = torch.max(labels2, dim=1)

    loss1 = loss_1(out1,labels1)
    loss2 = loss_2(out2,labels2)

    loss = loss1 + loss2

    correct_predictions1 += torch.sum(preds1 == org1)
    correct_predictions2 += torch.sum(preds2 == org2)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions1.double() / n_examples, correct_predictions2.double() / n_examples

def eval_model(model, data_loader, loss_1,loss_2, device, n_examples):

  model = model.eval()
  correct_predictions1 = 0
  correct_predictions2 = 0
  f_pred1 = []
  f_pred2 = []
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels1 = d["label1"].to(device)
      labels2 = d['label2'].to(device)
      out1, out2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds1 = torch.max(out1, dim=1)
      _, preds2 = torch.max(out2, dim=1)

      _, org1 = torch.max(labels1, dim=1)
      _, org2 = torch.max(labels2, dim=1)

      f_pred1.append(preds1)
      f_pred2.append(preds2)
        

      correct_predictions1 += torch.sum(preds1 == org1)
      correct_predictions2 += torch.sum(preds2 == org2)


  return correct_predictions1.double() / n_examples, correct_predictions2.double() / n_examples
	
	
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

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, y):
        #y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()