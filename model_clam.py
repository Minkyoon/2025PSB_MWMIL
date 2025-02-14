import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from transformers import BertModel


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),    
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)


    def forward(self, x):

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    
    

class MultiHeadAttention(nn.Module):
    def __init__(self, image_feature_dim, table_feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.image_attention = nn.ModuleList([nn.Linear(image_feature_dim, 1) for _ in range(num_heads)])
        self.table_attention = nn.ModuleList([nn.Linear(table_feature_dim, table_feature_dim) for _ in range(num_heads)])

    def forward(self, image_feature, table_feature):
        attended_features = []
        score =[]
        for i in range(self.num_heads):
            # image_feature에 대한 attention score 계산
            image_attention_score = torch.sigmoid(self.image_attention[i](image_feature))

        
            table_attention_score = torch.softmax(self.table_attention[i](table_feature), dim=1)

            
            attended_image_feature = image_attention_score * image_feature
            attended_table_feature = table_attention_score * table_feature
            
            score.append(table_attention_score)
            

            # attended_image_feature와 attended_table_feature를 합침
  
            concat_feature = torch.cat([attended_image_feature, attended_table_feature], dim=1)
            attended_features.append(concat_feature)

        # 모든 헤드의 결과를 연결
        multi_head_feature = torch.cat(attended_features, dim=1)
        return multi_head_feature, score


class Attention(nn.Module):
    def __init__(self, feature_dim, use_softmax=False):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.use_softmax = use_softmax
        self.attention = nn.Linear(feature_dim, feature_dim) if use_softmax else nn.Linear(feature_dim, 1)

    def forward(self, features):
        attention_scores = self.attention(features)
        if self.use_softmax:
            attention_scores = torch.softmax(attention_scores, dim=1)
        else:
            attention_scores = torch.sigmoid(attention_scores)
        attended_features = attention_scores * features
        return attended_features

class CLAM_endo(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=7, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=True):
        
        super(CLAM_endo, self).__init__()
        self.size_dict = {"small": [1024, 768, 512], "big": [1024, 512, 384]}
        self.size_arg = size_arg
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], 768), nn.ReLU()]
       
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.attention_module=MultiHeadAttention(768, 49,1 )
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample  
        self.transformer_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)
        

    def relocate(self):
        device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        self.attention_module=self.attention_module.to(device)
        self.transformer_encoder = self.transformer_encoder.to(device)

    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)       
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h,  label=None, instance_eval=True, return_features=False, attention_only=False):
        h=h.squeeze()
        A, h = self.attention_net(h)   
        A = torch.transpose(A, 1, 0)  
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  
        A_reshaped = A.squeeze(0).unsqueeze(1) 
 
        

        transformer_outputs = self.transformer_encoder(inputs_embeds=h.unsqueeze(0))[0]  # [1, k, hidden_size]
        aggregated_output= transformer_outputs *  h   * A_reshaped
        aggregated_output=aggregated_output.sum(dim=1)         

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

       

        return aggregated_output, A_raw, results_dict,
    
class CLAM_mre(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=7, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=True):
        
        super(CLAM_mre, self).__init__()
        self.size_dict = {"small": [1024, 768, 512], "big": [1024, 512, 384]}
        self.size_arg = size_arg
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], 768), nn.ReLU()]
       
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)
        

    def relocate(self):
        device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        self.attention_module=self.attention_module.to(device)
        self.transformer_encoder = self.transformer_encoder.to(device)

    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)  
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=True, return_features=False, attention_only=False):
        h=h.squeeze()
        A, h = self.attention_net(h)  # NxK    
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
            

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

       

        return M, A_raw, results_dict,  
    





class MultimodalModel(nn.Module):
    def __init__(self, endo_model, mre_model, instance_loss_fn=nn.CrossEntropyLoss()):
        super(MultimodalModel, self).__init__()
        self.endo_model = endo_model
        self.mre_model = mre_model
        self.classifier = nn.Sequential(
            nn.Linear(768*2+49, 768),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),  
            nn.Linear(128, 2),
        )
        self.endo_attention = Attention(768, use_softmax=False)  # Assuming 768 is the feature dim for endo
        self.mre_attention = Attention(768, use_softmax=False)  # Assuming 768 is the feature dim for MRE
        self.tabular_attention = Attention(49, use_softmax=True) 
         

    def forward(self, endo_data, mre_data, tabular_data, label=None):
        # endo
        endo_feat, A_raw_endo, results_dict_endo  = self.endo_model(endo_data, label=label)
        endo_feat = self.endo_attention(endo_feat)
        
        # MRE 
        mre_feat, A_raw_mre, results_dict_mre  = self.mre_model(mre_data, label=label)
        mre_feat = self.mre_attention(mre_feat)
        
        # tabualr    
        tabular_feat = self.tabular_attention(tabular_data)

        # fusion
        combined_features = self.late_fusion(endo_feat, mre_feat, tabular_feat)

        # fianl
        output = self.classifier(combined_features)
        return output, results_dict_endo, results_dict_mre


    def late_fusion(self, endo_features, mre_features, tabular_features):
        combined = torch.cat([endo_features, mre_features, tabular_features], dim=1)
        return combined




