"""Optional lazy PyTorch backend; importing the main app never imports torch."""
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_backend():
    import torch
    return torch.__version__

def network(features,hidden):
    import torch
    class SmallGRU(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.gru=torch.nn.GRU(features,hidden,num_layers=1,batch_first=True); self.output=torch.nn.Linear(hidden,1)
        def forward(self,x):
            _,state=self.gru(x); return self.output(state[-1]).squeeze(-1)
    return SmallGRU()

def seed(value):
    import torch
    np.random.seed(value); torch.manual_seed(value); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def fit(x,y,xval,yval,settings,random_seed):
    import torch
    seed(random_seed)
    scaler=StandardScaler().fit(x.reshape(-1,x.shape[-1]))
    transform=lambda v:torch.tensor(scaler.transform(v.reshape(-1,v.shape[-1])).reshape(v.shape),dtype=torch.float32)
    xt=transform(x); xv=transform(xval); yt=torch.tensor(y,dtype=torch.float32); yv=torch.tensor(yval,dtype=torch.float32)
    model=network(x.shape[-1],settings['hidden_units']); optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate']); criterion=torch.nn.BCEWithLogitsLoss()
    best=float('inf'); state=None; best_epoch=1; wait=0
    for epoch in range(settings['max_epochs']):
        model.train()
        for batch in torch.randperm(len(xt)).split(settings['batch_size']):
            optimizer.zero_grad(); loss=criterion(model(xt[batch]),yt[batch]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        model.eval()
        with torch.no_grad(): current=criterion(model(xv),yv).item()
        if current<best-1e-5:
            best=current; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; wait=0
        else: wait+=1
        if wait>=settings['patience']: break
    model.load_state_dict(state); artifact=pack(model,scaler,x.shape[-1],settings['hidden_units'])
    return artifact,predict(artifact,xval),best_epoch

def pack(model,scaler,features,hidden):
    return {'scaler':scaler,'state':{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},'features':features,'hidden':hidden}

def refit(x,y,settings,random_seed,epochs):
    import torch
    seed(random_seed); scaler=StandardScaler().fit(x.reshape(-1,x.shape[-1]))
    xt=torch.tensor(scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape),dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
    model=network(x.shape[-1],settings['hidden_units']); optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate']); criterion=torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for batch in torch.randperm(len(xt)).split(settings['batch_size']):
            optimizer.zero_grad(); loss=criterion(model(xt[batch]),yt[batch]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    return pack(model,scaler,x.shape[-1],settings['hidden_units'])

def predict(artifact,x):
    import torch
    model=network(artifact['features'],artifact['hidden']); model.load_state_dict({k:torch.tensor(v) for k,v in artifact['state'].items()}); model.eval()
    z=artifact['scaler'].transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
    with torch.no_grad(): return torch.sigmoid(model(torch.tensor(z,dtype=torch.float32))).numpy()
