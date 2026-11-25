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
        return self.ll2(F.relu(self.ll1(x)))
    
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
        self.expert_num=expert_num
        self.top_k=top_k

    def forward(self,x):
        router_logits=self.gate(x)
        router_prob=F.softmax(router_logits,dim=-1,dtype=torch.float)
        router_weights,selected_experts=torch.topk(router_prob,self.top_k,dim=-1)

        router_weights=router_weights/router_weights.sum(dim=-1,keepdim=True)
        router_weights=router_weights.to(x.dtype)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.expert_num
        )
        expert_mask = expert_mask.permute(2, 1, 0) 
        return router_logits, router_weights, selected_experts, expert_mask
        
class SparseMoEConfig:
    def __init__(self,experts,top_k,hidden_dim,embedding_dim,shared_expert=None,shared_expert_num=None):
        self.experts=experts
        self.top_k=top_k
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.shared_expert=shared_expert
        self.shared_expert_num=shared_expert_num
        

class SparseMOE(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.embedding_dim=config.embedding_dim
        self.hidden_dim=config.hidden_dim
        self.expert_num=config.experts
        self.top_k=config.top_k
        self.experts=nn.ModuleList(
            [
                MOE_Expert(self.embedding_dim,self.hidden_dim) for _ in range(self.expert_num)
            ]
        )

        self.router=SparseMOERouter(self.embedding_dim,self.expert_num,self.top_k)
        
    def forward(self, x):
        batch, seq_len, e_size = x.size()
        x_flat = x.view(-1, e_size)                                         # (T, E)

        router_logits, router_weights, selected_experts, expert_mask = self.router(x_flat)

        final_output = torch.zeros((batch * seq_len, e_size), dtype=x.dtype, device=x.device)

        for index in range(self.expert_num):
            expert_network = self.experts[index]
            idx, top_x = torch.where(expert_mask[index])                    # slot, token

            if top_x.numel() == 0:
                continue
            current_tokens = x_flat[top_x]                                  # (n_i, E)
            expert_out = expert_network(current_tokens)                     # (n_i, E)
            expert_out = expert_out * router_weights[top_x, idx].unsqueeze(-1)

            final_output.index_add_(0, top_x, expert_out.to(x.dtype))

        return final_output.view(batch, seq_len, e_size), router_logits

class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                MOE_Expert(config.embedding_dim, config.hidden_dim) for _ in range(config.shared_expert_num or 0)
            ]
        )
    def forward(self, x):
        sparse_moe_out, router_logits = self.moe_model(x)

        if len(self.shared_experts) == 0:
            return sparse_moe_out, router_logits

        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ]
        shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)   # (batch, seq_len, E)
        return sparse_moe_out + shared_experts_out, router_logits