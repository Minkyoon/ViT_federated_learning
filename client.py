import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import copy



def split_dataset_by_class(dataset, train_ratio, valid_ratio):
    # 클래스별로 데이터를 분할
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # 각 클래스별로 훈련, 검증, 테스트 세트로 분할
    train_indices, valid_indices, test_indices = [], [], []
    for label, indices in class_indices.items():
        train_idx, test_idx = train_test_split(indices, train_size=train_ratio + valid_ratio, random_state=42)
        valid_idx, test_idx = train_test_split(test_idx, train_size=valid_ratio / (1 - train_ratio - valid_ratio), random_state=42)
        train_indices.extend(train_idx)
        valid_indices.extend(valid_idx)
        test_indices.extend(test_idx)

    # 분할된 인덱스를 사용하여 Subset 생성
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, valid_subset, test_subset                


class Client:
    def __init__(self, client_id, dataset, model, lr, loss_fn, device):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.device = device

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                client_start = self.client_id * self.model.client_prompts
                client_end = client_start + self.model.client_prompts

        return self.model.prompts.data[client_start:client_end], self.model.vit.classifier.state_dict()

        
    
    def update_model(self, avg_prompts, avg_classifier):
        self.model.prompts.data = copy.deepcopy(avg_prompts)
        self.model.vit.classifier.load_state_dict(avg_classifier)
            



class Server:
    def __init__(self, model):
        self.model = model

    def aggregate(self, client_updates):
        # 각 클라이언트의 프롬프트를 적절한 위치에 배치
        for i, (client_prompts, _) in enumerate(client_updates):
            client_start = i * self.model.client_prompts
            client_end = client_start + self.model.client_prompts
            self.model.prompts.data[client_start:client_end] = client_prompts

        # 분류기 업데이트 집계
        classifier_updates = [update[1] for update in client_updates]
        avg_classifier = {key: torch.mean(torch.stack([update[key] for update in classifier_updates]), dim=0) for key in classifier_updates[0]}

        # 전체 모델 업데이트
        self.model.vit.classifier.load_state_dict(avg_classifier)

        return self.model.prompts.data, avg_classifier

    def distribute_model(self, clients, avg_prompts, avg_classifier):
        for client in clients:
            client.update_model(avg_prompts, avg_classifier)



