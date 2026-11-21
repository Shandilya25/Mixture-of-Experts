import torch 
import torch.nn as nn

class MOE_Expert(nn.Module):
    def __init__(self,feature_in,hidden_dim,feature_out):
        super().__init__()
        self.feature_in=feature_in
        self.feature_out=feature_out
        self.ll1=nn.Linear(feature_in,hidden_dim)
        self.ll2=nn.Linear(hidden_dim,feature_out)
        
    
    def forward(self,x):
        return self.ll2(nn.relu(self.ll1))
