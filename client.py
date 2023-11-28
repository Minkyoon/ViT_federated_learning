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
                
def partial_trim_attack(self):
        # 손상된 클라이언트의 그래디언트에 대한 평균과 표준편차 계산
        grads = [param.grad for param in self.model.parameters() if param.requires_grad]
        mean_grads = torch.mean(torch.stack(grads), dim=0)
        std_grads = torch.std(torch.stack(grads), dim=0, unbiased=False)

        # 그래디언트 조작
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    param.grad -= 3.5 * std_grads * torch.sign(mean_grads)
            


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
                if self.is_malicious:
                    self.partial_trim_attack()

                self.optimizer.step()
            print(f'클라이언트 {self.client_id}, 에폭 {epoch}')
            
        return self.model.state_dict()
            
    def partial_trim_attack(self):
        with torch.no_grad():
            flattened_grads = [param.grad.view(-1) for param in self.model.parameters() if param.requires_grad]
            

            all_grads = torch.cat(flattened_grads)  
            mean_grads = torch.mean(all_grads)
            std_grads = torch.std(all_grads)

            for param in self.model.parameters():
                if param.requires_grad:
                    param.grad -= 3.5 * std_grads * torch.sign(mean_grads)

        
