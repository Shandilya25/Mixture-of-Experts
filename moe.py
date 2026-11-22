import torch 
import torch.nn as nn
import torch.nn.functional as F


class MOE_Expert(nn.Module):
    def __init__(self,feature_in,hidden_dim):
        super().__init__()
        self.feature_in=feature_in
        self.ll1=nn.Linear(feature_in,hidden_dim)
        self.ll2=nn.Linear(hidden_dim,feature_in)
        
    
    def forward(self,x):
        return self.ll2(nn.relu(self.ll1))
    
class MOE(nn.Module):
    def __init__(self,experts,features_in,hidden_dim):
        self.expert_num=experts
        self.experts=nn.ModuleList(
            [
                MOE_Expert(features_in,hidden_dim) for _ in range(experts)
            ]
        )
        self.gate=nn.Linear(features_in,experts)
    
    def forward(self,x):
        #x.shape=(batch,seq_len,feature_in)--> expert_weight=(batch,seq_len,experts)
        expert_weights=self.gate(x)
        expert_weight = F.softmax(expert_weight, dim=-1)
        moe_out=[expert(x).unsqueeze(-2) for expert in self.experts]
        moe_out=torch.cat(moe_out,dim=-2)
        expert_weight = expert_weight.unsqueeze(-2)
        output = expert_weight @ moe_out 
        return output.squeeze()
        
        
#Sparse MOE Router 
class SparseMOERouter(nn.Module):
    def __init__(self,embedding_dim,expert_num,top_k):
        super().__init__()
        self.gate=nn.Linear(embedding_dim,expert_num)
        self.top_k=top_k
        
    def forward(self,x):
        router_logits=self.gate(x)
        router_prob=F.Softmax(router_logits,dim=-1,dtype=torch.float)
        router_weights=torch.topk(router_prob,self.top_k,dim=-1)
        