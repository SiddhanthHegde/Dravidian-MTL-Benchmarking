import pandas as pd
import torch
from torch.utils.data import Dataset

class MTLdataset(Dataset):

  def __init__(self,sentence,label1,label2,tokenizer,max_len):
    self.sentence = sentence
    self.label1 = label1
    self.label2 = label2
    self.tokenizer = tokenizer
    self.max_len = max_len
	
  def __len__(self):
    return len(self.sentence)	
	
  def __getitem__(self,item):
   
    sentence = str(self.sentence[item])
    label1 = self.label1[item]
    label2 = self.label2[item]

    encoding = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens = True,
      max_length = self.max_len,
      return_token_type_ids = False,
      padding ='max_length' , 
      return_attention_mask = True,
      return_tensors = 'pt',
      truncation = True
    )
    return {
      'sentences' : sentence,
      'input_ids' : encoding['input_ids'].flatten(),
      'attention_mask' : encoding['attention_mask'].flatten(),
      'label1' : torch.tensor(label1,dtype=torch.long),
      'label2' : torch.tensor(label2,dtype=torch.long)
    }