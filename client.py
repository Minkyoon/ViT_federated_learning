import torch
from torch.utils.data import DataLoader

# 기존의 add_malicious_updates 함수 정의 유지
def add_malicious_updates(model, noise_level=0.5, device='cpu'):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size()).to(device) * noise_level
            param.add_(noise)
            


def gaussian_attack(model, noise_level=0.1, device='cpu'):
    with torch.no_grad():  
        for param in model.parameters():
            if param.requires_grad:
                norm = param.grad.norm()
                noise = torch.randn_like(param.grad) * noise_level
                param.grad += noise * (norm / noise.norm())
                
                
def mean_attack(model):
    with torch.no_grad():  #
        for param in model.parameters():
            if param.requires_grad:
                param.grad.data = -param.grad.data
                
def partial_trim_attack(global_weights, f):
    # f: 손상된 클라이언트의 수
    # 손상된 클라이언트의 가중치만 선택
    malicious_weights = global_weights[:f]

    # 손상된 클라이언트의 가중치에 대한 평균과 표준편차 계산
    mean_weights = {key: torch.mean(torch.stack([weights[key] for weights in malicious_weights]), dim=0) for key in global_weights[0]}
    std_weights = {key: torch.std(torch.stack([weights[key] for weights in malicious_weights]), dim=0, unbiased=False) for key in global_weights[0]}

    # 공격 적용: 손상된 클라이언트의 가중치에만 공격 적용
    attacked_weights = []
    for weights in malicious_weights:
        attacked_weight = {key: weights[key] - 3.5 * std_weights[key] * torch.sign(mean_weights[key]) for key in weights}
        attacked_weights.append(attacked_weight)

    # 공격이 적용된 가중치와 나머지 클라이언트의 가중치를 결합
    return attacked_weights + global_weights[f:]
            
def full_trim_attack(global_weights, f):
    # f: 손상된 클라이언트의 수
    # 가중치 평균과 표준편차 계산
    mean_weights = {key: torch.mean(torch.stack([weights[key] for weights in global_weights]), dim=0) for key in global_weights[0]}
    std_weights = {key: torch.std(torch.stack([weights[key] for weights in global_weights]), dim=0, unbiased=False) for key in global_weights[0]}

    # 공격 적용
    for key in mean_weights.keys():
        mean_weights[key] = mean_weights[key] - 3.5 * std_weights[key] * torch.sign(mean_weights[key])

    return mean_weights


class Client:
    def __init__(self, client_id, dataset, model, lr, loss_fn, device, malicious_client_ids):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.device = device
        self.malicious_ids = malicious_client_ids
        self.is_malicious = client_id in malicious_client_ids  

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
            print(f'클라이언트 {self.client_id}, 에폭 {epoch}')
        
        # 악성 클라이언트일 경우 추가 업데이트 수행
        if self.is_malicious:
            print(f"poison:{self.client_id}")
            gaussian_attack(self.model, noise_level=20, device='cpu')

        
        return self.model.state_dict()
