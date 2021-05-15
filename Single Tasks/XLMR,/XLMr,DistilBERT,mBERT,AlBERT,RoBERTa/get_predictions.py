import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_predictions(model, data_loader):

  model = model.eval()
  sentence = []

  predictions = []
 
  prediction_probs = []
 
  real_values = []

  with torch.no_grad():
    for d in data_loader:
      texts = d["sentences"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels = d["label"].to(device)

      out1 = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      _, pred = torch.max(out1, dim=1)

      sentence.extend(texts)

      predictions.extend(pred)
      prediction_probs.extend(out1)
      real_values.extend(labels)

      
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()


  return sentence, predictions, prediction_probs, real_values