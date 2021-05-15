import pandas as pd
import torch
from torch.utils.data import Dataset

class MTLdataset(Dataset):

  def __init__(self,sentence,label,tokenizer,max_len):
    self.sentence = sentence
    self.label = label
    self.tokenizer = tokenizer
    self.max_len = max_len
	
  def __len__(self):
    return len(self.sentence)	
	
  def __getitem__(self,item):
   
    sentence = str(self.sentence[item])
    label = self.label1[item]

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
      'label' : torch.tensor(label,dtype=torch.long),
    }