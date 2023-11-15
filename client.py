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
