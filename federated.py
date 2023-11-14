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
import os





def add_malicious_updates(model, noise_level=0.5,device='cpu'):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size()).to(device) * noise_level  # 잡음을 올바른 디바이스로 이동
            param.add_(noise)
            
            

def is_malicious(client_id):

    return client_id in malicious_client_ids

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


#transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
])


# Hyperparameters
num_clients = 5
batch_size = 32
learning_rate = 1e-3
num_rounds = 10  
local_epochs = 10
malicious_client_ids = {0}  
poison_status = "mean"  
results_folder = "./results"  

# 결과 폴더가 존재하지 않으면 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 모델 성능 메트릭스 파일명 설정
metrics_filename = f'model_performance_metrics_round{num_rounds}_{poison_status}.txt'
conf_matrix_filename = f'confusion_matrix_round{num_rounds}_{poison_status}.png'



train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


data_size = len(train_dataset) // num_clients
client_datasets = [Subset(train_dataset, np.arange(i*data_size, (i+1)*data_size)) for i in range(num_clients)]


client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)







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




 


global_model = CustomViT()





global_model = global_model.to(device)


client_models = [CustomViT().to(device) for _ in range(num_clients)]
## deepcopy 사용하기

optimizers = [optim.SGD(client_model.parameters(), lr=learning_rate) for client_model in client_models]
## client class 만들기

loss_fn= nn.CrossEntropyLoss(reduction='sum')

# Training process

for round in range(num_rounds):
    client_weights = []
    for client_id, client_model in enumerate(client_models):
        client_model.train()  
        optimizer = optimizers[client_id]
        for epoch in range(local_epochs):
            for data, target in client_loaders[client_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
            print(f'클라이언트 {client_id}, 에폭 {epoch}')
         
        ## add noise in malicious clinets   
        if is_malicious(client_id):
            print(f"poison:{client_id}")
            add_malicious_updates(client_model, noise_level=0.1, device=device)    

      
        client_weights.append(client_model.state_dict())

    
    print(f'집계 라운드 {round}')
    global_weights = {key: torch.stack([client_weights[i][key] for i in range(num_clients)], 0).mean(0)
                      for key in client_weights[0]}
    
 
    global_model.load_state_dict(global_weights)

    
    for client_model in client_models:
        client_model.load_state_dict(global_model.state_dict())





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
plt.savefig(os.path.join(results_folder, conf_matrix_filename))



# Sensitivity 
sensitivity = recall_score(y_true, y_pred, average='macro')

# Accuracy 계산
accuracy = accuracy_score(y_true, y_pred)

class_report = classification_report(y_true, y_pred, target_names=train_dataset.classes)

# 결과를 txt 파일로 저장
with open(os.path.join(results_folder, metrics_filename), 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    # f.write(f'ROC AUC Score: {roc_auc}\n') 
    f.write(f'Sensitivity: {sensitivity}\n')
    f.write('\nClassification Report:\n')
    f.write(class_report)

print("Metrics saved to model_performance_metrics_round10_poison.txt")