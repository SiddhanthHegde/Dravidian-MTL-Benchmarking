import torch.nn as nn
from transformers import AutoModel
class MTLmodel(nn.Module):
	
  def __init__(self,n_classes,pretrained_model):
    super(MTLmodel,self).__init__()
    self.auto = AutoModel.from_pretrained(pretrained_model)
    self.drop = nn.Dropout(p=0.4)
    self.out1 = nn.Linear(self.auto.config.hidden_size,128)
    self.drop1 = nn.Dropout(p=0.4)
    self.relu = nn.ReLU()
    self.sent = nn.Linear(128,n_classes)
		
  def forward(self, input_ids, attention_mask):

    output_1 = self.auto(input_ids=input_ids, attention_mask=attention_mask)
    hidden_state = output_1[0]
    pooler = hidden_state[:, 0]
    pooler = self.out1(pooler)
    pooler = nn.ReLU()(pooler)
    pooler = self.drop1(pooler)
    return self.sent(pooler)