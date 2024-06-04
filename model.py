import torch
import torch.optim as optim
import torchmetrics
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import warnings

warnings.filterwarnings("ignore", message=r"Some weights of.*classifier.*are newly initialized")

class TravelDocClassifier:
  def __init__(self, device, params_path=None):
    self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    if params_path is not None:
      self.model.load_state_dict(torch.load(params_path))

    self.device = device

  def save_params(self, best_acc):
    current_date = datetime.now().strftime("%Y%m%d")
    file_path = f'parameters/{current_date}_{best_acc}_model_parameters.pth'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    torch.save(self.model.state_dict(), file_path)
    print(f'Saving parameters at {file_path}')

  def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=5e-5, patience=2):
    self.model.to(self.device)

    optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    min_val_loss = float('inf')
    no_improve_count = 0
    epoch_statistics = {'train_losses': [], 'val_losses': [], 'val_accuracies': [], 'train_accuracies': []}

    for epoch in range(epochs):
      self.model.train()
      train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(self.device)
      train_loss = torchmetrics.MeanMetric().to(self.device)
        
      for batch in train_loader:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.update(loss)
        train_acc.update(outputs.logits, batch['labels'])

      loss = train_loss.compute().item()
      acc = train_acc.compute().item() * 100
      epoch_statistics['train_losses'].append(loss)
      epoch_statistics['train_accuracies'].append(acc)
      train_acc.reset()
      train_loss.reset()

      print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss:.4f} | Train Accuracy: {acc:.2f}%")

      # Validation step
      self.model.eval()
      val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(self.device)
      val_loss_metric = torchmetrics.MeanMetric().to(self.device)

      with torch.no_grad():
        for batch in val_loader:
          batch = {k: v.to(self.device) for k, v in batch.items()}
          outputs = self.model(**batch)
          loss = outputs.loss
          val_loss_metric.update(loss)
          val_acc.update(outputs.logits, batch['labels'])

      val_loss = val_loss_metric.compute().item()
      val_accuracy = val_acc.compute().item() * 100
      epoch_statistics['val_losses'].append(val_loss)
      epoch_statistics['val_accuracies'].append(val_accuracy)

      if val_loss < min_val_loss:
        min_val_loss = val_loss
        no_improve_count = 0
        epoch_statistics['best_val_loss'] = val_loss
        epoch_statistics['best_val_accuracy'] = val_accuracy
      else:
        no_improve_count += 1

      if no_improve_count >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
    
    try:
      best_acc = int(epoch_statistics["best_val_accuracy"]*100)
      self.save_params(best_acc)
    except Exception as e:
      print(f"An error occurred: {e}")
      
    return epoch_statistics

  def test(self, test_dataset, batch_size=16):
    self.model.to(self.device)
    self.model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    predictions = []
    true_labels = []
      
    with torch.no_grad():
      for batch in test_loader:
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        pred_labels = torch.argmax(logits, axis=1)
        predictions.extend(pred_labels.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
  
    return accuracy, cm

  def predict(self, text):
    self.model.to(self.device)
    self.model.eval()

    text = {k: v.to(self.device) for k, v in text.items()}

    with torch.no_grad():
      outputs = self.model(**text)
      logits = outputs.logits
      pred_label = torch.argmax(logits, axis=1).cpu().numpy()[0]

    return pred_label
  
  def predict_all(self, texts):
    self.model.to(self.device)
    self.model.eval()

    dataloader = DataLoader(texts, batch_size=32)

    predictions = []

    with torch.no_grad():
      for batch in dataloader:
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        pred_labels = torch.argmax(logits, axis=1)
        predictions.extend(pred_labels.cpu().numpy())
    
    return predictions

  
  def plot_metrics(self, stats):
    epochs = len(stats['train_losses'])
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), stats['train_losses'], label='Train Loss')
    plt.plot(range(epochs), stats['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), stats['train_accuracies'], label='Train Accuracy')
    plt.plot(range(epochs), stats['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
