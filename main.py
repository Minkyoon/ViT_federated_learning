import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, ViTFeatureExtractor,ViTConfig, ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, classification_report
from scipy.stats import hmean
from client import Client  

# Hyperparameters
num_clients = 5
batch_size = 32
learning_rate = 1e-3
num_rounds = 10  
local_epochs = 10
malicious_client_ids = {}  # 0번째 클라이언트는 악성


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


#transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
])


# ViT 모델 정의
class CustomViT(nn.Module):
    def __init__(self, num_labels=10):
        super(CustomViT, self).__init__()
        configuration = ViTConfig(
            image_size=28,
            patch_size=7,
            num_channels=1,
            num_labels=num_labels,
            hidden_size=256,   
            num_hidden_layers=4,   
            num_attention_heads=8,
            intermediate_size=512  
        )
        self.vit = ViTForImageClassification(configuration)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits



# 데이터셋 정의
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


#train dataset을 client수로 분할
data_size = len(train_dataset) // num_clients
client_datasets = [Subset(train_dataset, np.arange(i*data_size, (i+1)*data_size)) for i in range(num_clients)]


# 글로벌 모델 선언
global_model =  CustomViT().to(device)
global_optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)

clients = [Client(client_id=i, 
                  dataset=client_datasets[i], 
                  model=CustomViT().to(device), 
                  lr=learning_rate,
                  loss_fn=nn.CrossEntropyLoss(reduction='sum'), 
                  device=device,
                  malicious_client_ids= malicious_client_ids)
           for i in range(num_clients)]

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




loss_fn=nn.CrossEntropyLoss(reduction='sum')





# 훈련 루프
for round in range(num_rounds):
    global_weights = []

    # 클라이언트별로 훈련 진행
    for client in clients:
        client_state_dict = client.train(local_epochs)
        global_weights.append(client_state_dict)
    
    # 글로벌 모델 가중치 업데이트
    new_global_state_dict = {key: torch.mean(
        torch.stack([client_weights[key] for client_weights in global_weights]), dim=0)
        for key in global_weights[0]}
    
    global_model.load_state_dict(new_global_state_dict)

    # 클라이언트 모델들을 글로벌 모델로 업데이트
    for client in clients:
        client.model.load_state_dict(global_model.state_dict())

    # 테스트셋으로 평가
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f'Round {round}: Accuracy on test set: {100 * correct / total:.2f}%')





global_model.eval() 
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device) 
        output = global_model(data)
        test_loss += loss_fn(output, target).item()  
        pred = output.argmax(dim=1, keepdim=True)  
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
plt.savefig('confusion_matrix.png')



# Sensitivity 
sensitivity = recall_score(y_true, y_pred, average='macro')

# Accuracy 계산
accuracy = accuracy_score(y_true, y_pred)

class_report = classification_report(y_true, y_pred, target_names=train_dataset.classes)

# 결과를 txt 파일로 저장
with open('model_performance_metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    # f.write(f'ROC AUC Score: {roc_auc}\n') 
    f.write(f'Sensitivity: {sensitivity}\n')
    f.write('\nClassification Report:\n')
    f.write(class_report)

