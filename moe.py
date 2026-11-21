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
    
class MOE(nn.Module):
    def __init__(self,experts,features_in,features_out,hidden_dim):
        self.expert_num=experts
        self.experts=nn.ModuleList(
            [
                MOE_Expert(features_in,hidden_dim,features_out) for _ in range(experts)
            ]
        )
        self.gate=nn.Linear(features_in,experts)
    
    def forward(self,x):
        #x.shape=(batch,seq_len,feature_in)--> expert_weight=(batch,seq_len,experts)
        expert_weights=self.gate(x)
        expert_weight = torch.softmax(expert_weight, dim=-1)
        moe_out=[expert(x).unsqueeze(-2) for expert in self.experts]
        moe_out=torch.cat(moe_out,dim=-2)
        expert_weight = expert_weight.unsqueeze(-2)
        output = expert_weight @ moe_out 
        return output.squeeze()
        
        
