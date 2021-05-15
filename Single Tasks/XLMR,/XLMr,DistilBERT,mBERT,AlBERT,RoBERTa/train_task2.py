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
from transformers import AdamW,AutoTokenizer
from collections import defaultdict
from get_predictions import get_predictions 



df = pd.read_csv('Final_Tamil_Dataset.csv')
df['off'], uniq1 = pd.factorize(df['off'])

train,val,test =np.split(df.sample(frac=1, random_state=42),[int(.8*len(df)), int(.9*len(df))])

pretrained_models = ['distilbert-base-multilingual-cased','bert-base-multilingual-cased','roberta-base','albert-base-v1','xlm-roberta-base'] 
#pretrained_model_kan_mal = 'bert-base-multilingual-cased'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in pretrained_models]

BATCH_SIZE = 16
MAX_LEN = 128

classification_reports = []
for tokenizer,pretrained_model in zip(tokenizers,pretrained_models):
    train_data_loader = create_data_loader(train,tokenizer,MAX_LEN,BATCH_SIZE)
    val_data_loader = create_data_loader(val,tokenizer,MAX_LEN,BATCH_SIZE,shuffle=False)
    test_data_loader = create_data_loader(test,tokenizer,MAX_LEN,BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss.to(device)

    model = MTLmodel(len(uniq1),pretrained_model)

    model = model.to(device)

    EPOCHS = 5
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):

        start_time = time.time()
        train_acc1,train_acc2 = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
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
        model,
        val_data_loader,
        loss_fn,
        device,
        val.shape[0] 
        )
        print(f'Val Acc1 {val_acc1} Val Acc2 {val_acc2}')

    y_review_texts, y_pred1, y_pred_probs1, y_test1 = get_predictions(
    model,
    test_data_loader
    )

    classification_reports.append(classification_report(y_test1, y_pred1, target_names=uniq1,zero_division=0))

    


