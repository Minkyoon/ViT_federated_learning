import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, ViTFeatureExtractor,ViTConfig, ViTForImageClassification
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset,random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, classification_report
from scipy.stats import hmean
from client import Client, Server, split_dataset_by_class  
import os
import random
from transformers import ViTForImageClassification
from sklearn.model_selection import train_test_split
import copy




class CustomViTWithPrompt(nn.Module):
    def __init__(self, num_labels, num_prompts):
        super(CustomViTWithPrompt, self).__init__()
        configuration = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_labels)
        self.vit = ViTForImageClassification(configuration)
        self.prompts = nn.Parameter(torch.randn(num_prompts, configuration.hidden_size))
        self.freeze_vit_parameters()

    def freeze_vit_parameters(self):
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        embeddings = self.vit.vit.embeddings(pixel_values)
        extended_embeddings = torch.cat([embeddings[:, 0:1], self.prompts.unsqueeze(0).expand(embeddings.size(0), -1, -1), embeddings[:, 1:]], dim=1)
        outputs = self.vit.vit.encoder(extended_embeddings)
        logits = self.vit.classifier(outputs[0][:, 0])
        return logits
    

# Hyperparameters
num_clients = 5
batch_size = 32
learning_rate = 1e-3
num_rounds = 50  
local_epochs = 5
malicious_client_ids = {0,1}  
poison_status = "trim_real"  
 



# 모델 성능 메트릭스 파일명 설정


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT에 맞는 크기 조정
    transforms.ToTensor(),          # 이미지를 PyTorch 텐서로 변환
    # 추가적인 변환들을 여기에 포함시킬 수 있습니다.
])









# 데이터셋 정의
# CIFAR-10 데이터셋 불러오기
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 데이터셋 분할
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

# 클라이언트별 데이터셋 분할
client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients + (1 if x < len(train_dataset) % num_clients else 0) for x in range(num_clients)])

# 클라이언트 서브셋 선택
selected_subsets = random.sample(client_datasets, num_clients)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

global_model = CustomViTWithPrompt(num_labels=10, num_prompts=50 ).to(device)
server = Server(global_model)

# Assuming you have a way to split your dataset into subsets for each client


clients = [Client(i, selected_subsets[i], copy.deepcopy(CustomViTWithPrompt(num_labels=10, num_prompts=50)).to(device), lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device) for i in range(num_clients)]



test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




loss_fn=nn.CrossEntropyLoss(reduction='sum')





# 훈련 루프
# 각 라운드별 성능 평가를 위한 리스트
round_accuracies = []
results_folder = './results/cifar_50ronud_5epoch_5client'
accuracy_log_filename = os.path.join(results_folder, 'round_accuracies.txt')

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

for round in range(num_rounds):
    client_updates = [client.train(epochs=local_epochs) for client in clients]
    avg_prompts, avg_classifier = server.aggregate(client_updates)
    server.distribute_model(clients, avg_prompts, avg_classifier)

    # 글로벌 모델 성능 평가 (정확도만 계산)
    global_model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            y_pred.extend(predicted.view(-1).cpu().numpy())
            y_true.extend(target.view(-1).cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Round {round}: Accuracy on test set: {accuracy:.2f}%')

    with open(accuracy_log_filename, 'a') as log_file:
        log_file.write(f'Round {round}: Accuracy on test set: {accuracy:.2f}%\n')
    


# 마지막 라운드의 혼동 행렬 및 기타 메트릭스 계산
conf_mat = confusion_matrix(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred, average='macro')
class_report = classification_report(y_true, y_pred, target_names=class_names)

# 혼동 행렬 시각화 및 저장
conf_matrix_filename = 'confusion_matrix.png'
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_folder, conf_matrix_filename))
plt.close(fig)

# 성능 메트릭스 저장
metrics_filename = 'performance_metrics.txt'
with open(os.path.join(results_folder, metrics_filename), 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Sensitivity: {sensitivity:.4f}\n')
    f.write('\nClassification Report:\n')
    f.write(class_report)


# 최종 모델 저장
model_save_path = os.path.join(results_folder, 'final_model.pth')
torch.save(global_model.state_dict(), model_save_path)
