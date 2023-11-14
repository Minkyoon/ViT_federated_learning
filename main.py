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
import os

# Hyperparameters
num_clients = 5
batch_size = 32
learning_rate = 1e-3
num_rounds = 10  
local_epochs = 5
malicious_client_ids = {0,1,2,3,4}  
poison_status = "gausian_noise_5noise"  
results_folder = "./results"  

# 결과 폴더가 존재하지 않으면 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 모델 성능 메트릭스 파일명 설정
metrics_filename = f'model_performance_metrics_round{num_rounds}_epoch{local_epochs}_{poison_status}.txt'
conf_matrix_filename = f'confusion_matrix_round{num_rounds}_{local_epochs}_{poison_status}.png'


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



# 예측값과 실제값을 저장할 리스트 초기화
y_pred = []
y_true = []

# 모델을 평가 모드로 설정
global_model.eval()

# 그라디언트 계산을 비활성화
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        # 모델로부터 예측값을 얻음
        outputs = global_model(data)
        # 가장 높은 값을 가진 인덱스를 예측값으로 선택
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.view(-1).cpu().numpy())
        y_true.extend(target.view(-1).cpu().numpy())

# 혼동 행렬 계산
conf_mat = confusion_matrix(y_true, y_pred)

# 혼동 행렬을 시각화
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.ylabel('actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.savefig(os.path.join(results_folder, conf_matrix_filename))

# 감도(Sensitivity) 계산
sensitivity = recall_score(y_true, y_pred, average='macro')

# 정확도(Accuracy) 계산
accuracy = accuracy_score(y_true, y_pred)

# 분류 보고서 생성
class_report = classification_report(y_true, y_pred, target_names=train_dataset.classes)

# 성능 메트릭스를 텍스트 파일로 저장
with open(os.path.join(results_folder, metrics_filename), 'w') as f:
    f.write(f'정확도(Accuracy): {accuracy:.4f}\n')
    f.write(f'감도(Sensitivity): {sensitivity:.4f}\n')
    f.write('\n분류 보고서(Classification Report):\n')
    f.write(class_report)

