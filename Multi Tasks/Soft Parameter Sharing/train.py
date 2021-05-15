import time
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
import random
from model import MTLmodel
from utils import create_data_loader,train_epoch,eval_model,FocalLoss,epoch_time
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup,AutoModel,AutoTokenizer
from collections import defaultdict
from get_predictions import get_predictions 



df = pd.read_csv('Final_Tamil_Dataset.csv')
df['sent'], uniq1 = pd.factorize(df['sent'])
df['off'], uniq2 = pd.factorize(df['off'])

train,val,test =np.split(df.sample(frac=1, random_state=42),[int(.8*len(df)), int(.9*len(df))])

pretrained_model_tamil = 'distilbert-base-multilingual-cased'
#pretrained_model_kan_mal = 'bert-base-multilingual-cased'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PRE_TRAINED_MODEL_NAME = 'distilbert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

BATCH_SIZE = 16
MAX_LEN = 128
train_data_loader = create_data_loader(train,tokenizer,MAX_LEN,BATCH_SIZE)
val_data_loader = create_data_loader(val,tokenizer,MAX_LEN,BATCH_SIZE,shuffle=False)
test_data_loader = create_data_loader(test,tokenizer,MAX_LEN,BATCH_SIZE)

focal_loss = FocalLoss().to(device)
ce_loss = nn.CrossEntropyLoss.to(device)
kld_loss = nn.KLDivLoss().to(device)
hinge_loss = nn.MultiLabelMarginLoss.to(device)

losses = [focal_loss,ce_loss,kld_loss,hinge_loss]

model1 = MTLmodel(len(uniq1),pretrained_model_tamil)
model2 = MTLmodel(len(uniq2),pretrained_model_tamil)

model1 = model1.to(device)
model2 = model2.to(device)

EPOCHS = 5
optimizer1 = AdamW(model1.parameters(), lr=2e-5, correct_bias=False)
optimizer2 = AdamW(model2.parameters(), lr=2e-5, correct_bias=False)

classification_reports_1 = []
classification_reports_2 = []

history = defaultdict(list)
best_accuracy = 0
for loss in losses:
    for epoch in range(EPOCHS):

        start_time = time.time()
        train_acc1,train_acc2 = train_epoch(
            model1,
            model2,
            train_data_loader,
            loss,
            loss,
            optimizer1,
            optimizer2,
            device,
            train.shape[0]
        )
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Acc1 {train_acc1} Train Acc2 {train_acc2}')
        print()

        history['train_acc1'].append(train_acc1)
        history['train_acc2'].append(train_acc2)

        val_acc1, val_acc2 = eval_model(
        model1,
        model2,
        val_data_loader,
        loss,
        loss,
        device,
        val.shape[0] 
        )
        print(f'Val Acc1 {val_acc1} Val Acc2 {val_acc2}')

    y_review_texts, y_pred1, y_pred_probs1, y_test1, _, _, _  = get_predictions(
    model1,
    model2,
    test_data_loader
    )

    y_review_texts, _, _, _,  y_pred2, y_pred_probs2, y_test2 = get_predictions(
    model1,
    model2,
    test_data_loader
    )

    classification_reports_1.append(classification_report(y_test1, y_pred1, target_names=uniq1,zero_division=0))
    classification_reports_2.append(classification_report(y_test2, y_pred2, target_names=uniq2,zero_division=0))

print(classification_reports_1)
print(classification_reports_1)

    


