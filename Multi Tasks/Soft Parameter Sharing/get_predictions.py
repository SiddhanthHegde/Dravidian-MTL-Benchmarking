import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_predictions(model1,model2, data_loader):

  model1 = model1.eval()
  model2 = model2.eval()
  sentence = []

  predictions1 = []
  predictions2 = []
 
  prediction_probs1 = []
  prediction_probs2 = []
 
  real_values1 = []
  real_values2 = []

  with torch.no_grad():
    for d in data_loader:
      texts = d["sentences"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels1 = d["label1"].to(device)
      labels2 = d["label2"].to(device)
      out1 = model1(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      out2 = model2(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, pred1 = torch.max(out1, dim=1)
      _,pred2 = torch.max(out2,dim=1)

      sentence.extend(texts)

      predictions1.extend(pred1)
      prediction_probs1.extend(out1)
      real_values1.extend(labels1)

      predictions2.extend(pred2)
      prediction_probs2.extend(out2)
      real_values2.extend(labels2)
      
  predictions1 = torch.stack(predictions1).cpu()
  prediction_probs1 = torch.stack(prediction_probs1).cpu()
  real_values1 = torch.stack(real_values1).cpu()

  predictions2 = torch.stack(predictions2).cpu()
  prediction_probs2 = torch.stack(prediction_probs2).cpu()
  real_values2 = torch.stack(real_values2).cpu()

  return sentence, predictions1, prediction_probs1, real_values1, predictions2, prediction_probs2, real_values2