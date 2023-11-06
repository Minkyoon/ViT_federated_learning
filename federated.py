import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor,ViTConfig, ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, classification_report
from scipy.stats import hmean



device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# 데이터셋 변환 정의
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # ViT 모델에 맞게 이미지 크기를 조정합니다.
    transforms.ToTensor(),
    
])


# Hyperparameters
num_clients = 5
local_epochs = 100
batch_size = 32
learning_rate = 1e-3



train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# FashionMNIST 데이터셋을 여러 클라이언트로 분할
data_size = len(train_dataset) // num_clients
client_datasets = [Subset(train_dataset, np.arange(i*data_size, (i+1)*data_size)) for i in range(num_clients)]

# 클라이언트 DataLoader 정의
client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]

# 테스트 데이터셋 DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# ... [데이터셋 로딩 코드] ...



# ViT 모델 정의
class CustomViT(nn.Module):
    def __init__(self, num_labels=10):
        super(CustomViT, self).__init__()
        configuration = ViTConfig(
            image_size=28,
            patch_size=7,
            num_channels=1,
            num_labels=num_labels,
            hidden_size=256,   # 히든 레이어 크기는 조정 가능
            num_hidden_layers=4,   # Transformer 레이어 수도 조정 가능
            num_attention_heads=8,
            intermediate_size=512  # Feedforward 레이어의 크기
        )
        self.vit = ViTForImageClassification(configuration)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits




 

# 모델 초기화
global_model = CustomViT()

# 모델 초기화


# Move the global model to GPU if available
global_model = global_model.to(device)

# Initialize and move client models to GPU if available
client_models = [CustomViT().to(device) for _ in range(num_clients)]


# Initialize client models
for client_model in client_models:
    client_model.load_state_dict(global_model.state_dict())

# Define the optimizers for each client model
optimizers = [optim.SGD(client_model.parameters(), lr=learning_rate) for client_model in client_models]

loss_fn= nn.CrossEntropyLoss(reduction='sum')

# Training process
for epoch in range(local_epochs):
    # List to collect client model weights after training
    client_weights = []
    for client_id, client_model in enumerate(client_models):
        client_model.train()  # Set the model to training mode
        optimizer = optimizers[client_id]
        for data, target in client_loaders[client_id]:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            print(epoch)

        # Append client model weights to the list
        client_weights.append(client_model.state_dict())

    # Update global weights as the average of client model weights
    
    print(f'aggre{epoch}')
    global_weights = {key: torch.stack([client_weights[i][key] for i in range(num_clients)], 0).mean(0)
                      for key in client_weights[0]}

    # Update global model
    global_model.load_state_dict(global_weights)

    # Optionally, you can now update each client model to the new global model
    for client_model in client_models:
        client_model.load_state_dict(global_model.state_dict())

# Evaluate the global model on the test dataset
global_model.eval()  # Set the model to evaluation mode
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device) 
        output = global_model(data)
        test_loss += loss_fn(output, target).item()  # Sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
test_accuracy = 100. * correct / len(test_loader.dataset)

print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)')


y_pred = []
y_true = []
global_model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        outputs = global_model(data)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.view(-1).cpu().numpy())
        y_true.extend(target.view(-1).cpu().numpy())

# Confusion Matrix 계산
conf_mat = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix2.png')



# Sensitivity (recall) 계산
sensitivity = recall_score(y_true, y_pred, average='macro')

# Accuracy 계산
accuracy = accuracy_score(y_true, y_pred)

# Classification report (precision, recall, f1-score)
class_report = classification_report(y_true, y_pred, target_names=train_dataset.classes)

# 결과를 txt 파일로 저장
with open('model_performance_metrics2.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    # f.write(f'ROC AUC Score: {roc_auc}\n') # 이진 분류인 경우에만 주석 해제
    f.write(f'Sensitivity: {sensitivity}\n')
    f.write('\nClassification Report:\n')
    f.write(class_report)

print("Metrics saved to model_performance_metrics.txt")