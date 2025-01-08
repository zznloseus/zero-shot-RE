import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForPreTraining, AlbertForPreTraining, DebertaV2PreTrainedModel, DebertaV2ForMaskedLM
from dataset import pad_or_truncate
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


# pooling
def extract_entity(sequence_output, e_mask):
    # print(e_mask.size())   # [32, 128]
    extended_e_mask = e_mask.unsqueeze(-1)
    # print(extended_e_mask.size()) # [32, 128, 1]
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    # print(extended_e_mask.size()) # [32, 768]
    # extended_e_mask = torch.stack([sequence_output[i,j,:] for i,j in enumerate(e_mask)])
    # print(extended_e_mask)
    return extended_e_mask.float()


class EMMA(nn.Module):  
    def __init__(self, pretrain_model_name_or_path, add_auto_match=True, max_seq_len=128, k=3):
        super(EMMA, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.bert = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.k = k
        self.cos = nn.CosineSimilarity(dim=-1)
        # self.alpha = nn.Parameter(torch.tensor(0.3))
        self.add_auto_match = add_auto_match
        # if add_auto_match:
        #     self.des_weights1 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
        #     self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
        #     self.des_weights2 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
        #     self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
        #     # self.sen_att_cls = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        #     # self.sen_att_head = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        #     # self.sen_att_tail = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        #     # self.sen_att = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        
        self.des_e1 = nn.Parameter(torch.ones(self.config.hidden_size, self.config.hidden_size))

        self.des_e2 = nn.Parameter(torch.ones(self.config.hidden_size, self.config.hidden_size))
        
        self.des_en = nn.Parameter(torch.ones(self.config.hidden_size, self.config.hidden_size))
        
        self.sen_feature = nn.Linear(3*self.config.hidden_size, self.config.hidden_size)
        # self.e1_entity_mlp = nn.Parameter(torch.ones(2*self.config.hidden_size, self.config.hidden_size))
        # self.e2_entity_mlp = nn.Parameter(torch.ones(2*self.config.hidden_size, self.config.hidden_size))

        if add_auto_match:
            self.des_weights1 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            # self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
            self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len, 1))
            self.des_weights2 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            # self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
            self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len, 1))
            self.des_bias3 = nn.Parameter(torch.zeros(max_seq_len, 1))
        
        self.classify = Classify_model(pretrain_model_name_or_path, max_seq_len, k)
        


        self.des_vectors = None
    
    
    # def forward(self, sen_input_ids, sen_att_masks, des_input_ids, des_att_masks, sen_e1_pos, sen_e2_pos, sen_e1_pos_end, sen_e2_pos_end):
    def forward(self, sen_input_ids, sen_att_masks, des_input_ids=None, des_att_masks=None, \
                marked_e1=None, marked_e2=None, mark_head=None, mark_tail=None, head_range=None, tail_range=None, neg_counts=1):
        '''
        sen_input_ids: [bs, max_seq_length]
        sen_att_masks: [bs, max_seq_length]
        des_input_ids: [bs, max_seq_length]
        des_att_masks: [bs, max_seq_length]
        marked_e1: [bs, max_seq_length] 
        marked_e2: [bs, max_seq_length]
        mark_head: [bs, max_seq_length]
        mark_tail: [bs, max_seq_length]
        '''
        batch_size = sen_input_ids.size(0)
        device = sen_input_ids.device
        
        # sentence = [cls] + ... + [E1] + E1 + [E1/] + ... + [E2] + E2 + [E2/] + .....
        # label_description = [cls] + .....

        sen_outputs = self.bert(
            input_ids=sen_input_ids,
            attention_mask=sen_att_masks,
        )
        sen_output = sen_outputs.last_hidden_state
        # sen_cls = torch.mean(sen_output,dim=1) 
        sen_e1_entity_vec, sen_e2_entity_vec = self.get_sen_entity_vec(sen_output, marked_e1, marked_e2)
        
        sen_entity_between_vec = self.get_sen_entity_between_vec(sen_output, marked_e1, marked_e2)

        # entity_vec = torch.cat([sen_e1_entity_vec, sen_e2_entity_vec],dim=1)

        sen_entity_vec = F.normalize(torch.cat([sen_e1_entity_vec,sen_e2_entity_vec,sen_entity_between_vec],dim=1),p=2,dim=1) #[bs,2*hs]
        
        if des_input_ids != None:
            des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_att_masks,
            )
            des_output = des_outputs.last_hidden_state

            # des_vec = self.get_des_vec(des_output)

            
        # [cls] xxx xxx xxx [E1] head_entity [E1/] xxx xxx [E2] tail_entity [E2/] xxx xxx xxx
        # e1_h = extract_entity(sen_output, marked_e1) # [E1]
        # e2_h = extract_entity(sen_output, marked_e2) # [E2]
        # head_entity = extract_entity(sen_output, mark_head) # head_entity
        # tail_entity = extract_entity(sen_output, mark_tail) # tail_entity
        
        # train
        if des_input_ids != None:
            # cos_sim_ctx = self.cos(context_proj.unsqueeze(1), train_des_features[:,0,:].unsqueeze(0)) # [32, 1, 768] # [1, 32, 768]
            # cos_sim_e1 = self.cos(e1_proj.unsqueeze(1), train_des_features[:,1,:].unsqueeze(0))
            # cos_sim_e2 = self.cos(e2_proj.unsqueeze(1), train_des_features[:,2,:].unsqueeze(0)) 
            # integrated_sim = self.alpha*(cos_sim_e1 + cos_sim_e2) + (1-self.alpha)*cos_sim_ctx

            #聚合步骤
            # print(torch.einsum('bmi,ij,bqj->bmq',des_output, self.des_e1, sen_e1_entity_vec.unsqueeze(1)).shape)
            # sen_e1_feature = self.e1_entity_mlp(torch.cat([sen_e1_entity_vec,sen_entity_between_vec],dim=-1))
            des_e1_layer = torch.squeeze(torch.add(torch.einsum('bmi,ij,bqj->bmq',des_output, self.des_e1, sen_e1_entity_vec.unsqueeze(1)),self.des_bias1),dim=-1) #[bs,ml,ml]
            des_e1_layer1_softmax = torch.softmax(des_e1_layer,dim=1)
            des_e1_vec = torch.sum(torch.unsqueeze(des_e1_layer1_softmax, dim=2) * des_output, dim=1)
            
            # sen_e2_feature = self.e2_entity_mlp(torch.cat([sen_e2_entity_vec,sen_entity_between_vec],dim=-1))
            des_e2_layer = torch.squeeze(torch.add(torch.einsum('bmi,ij,bqj->bmq',des_output, self.des_e2, sen_e2_entity_vec.unsqueeze(1)),self.des_bias2),dim=-1)#[bs,ml,ml]
            des_e2_layer2_softmax = torch.softmax(des_e2_layer,dim=1)
            des_e2_vec = torch.sum(torch.unsqueeze(des_e2_layer2_softmax, dim=2) * des_output, dim=1)
            
            des_en_layer = torch.squeeze(torch.add(torch.einsum('bmi,ij,cqj->bcmq',des_output, self.des_en, sen_entity_between_vec.unsqueeze(1)),self.des_bias3),dim=-1)
            # des_bs_en_weight, _ = torch.max(des_en_layer, dim=1) #[bs_des,des_ml,des_ml]
            des_en_layer_softmax = torch.softmax(des_bs_en_weight,dim=1) #[bs,ml]
            des_en_vec = torch.sum(torch.unsqueeze(des_en_layer_softmax, dim=-1) * des_output, dim=1)

               

            des_entity_vec = F.normalize(torch.cat([des_e1_vec,des_e2_vec,des_en_vec],dim=1),p=2,dim=1) #[bs,2*hs]

            # sen_vec_feature = self.sen_feature_mlp(torch.cat([torch.mean(sen_output,dim=1),sen_entity_between_vec],dim=-1))
            # des_vec_feature = self.des_feature_mlp(torch.cat([torch.mean(des_output,dim=1),des_entity_between_vec],dim=-1))
            # sen_vec_feature = self.sen_feature(torch.cat([sen_cls, sen_e1_entity_vec, sen_e2_entity_vec],dim=-1)) 
            sen_vec_feature = torch.mean(sen_output,dim=1)           
            des_vec_feature = torch.mean(des_output,dim=1)
            
            # sen_entity_between_vec

            des_vec = torch.cat([des_vec_feature ,des_entity_vec], dim=1)
            sen_vec = torch.cat([sen_vec_feature, sen_entity_vec], dim=1) #[bs,3*bs]
            
            # des_sim = self.cos(des_input_ids.unsqueeze(1),des_input_ids.unsqueeze(0))
            # print(des_sim[0])
            feature_cos_sim = self.cos(sen_vec_feature.unsqueeze(1),des_vec_feature.unsqueeze(0))
            topk_values, feature_topk_indices = torch.topk(feature_cos_sim, k=self.k+neg_counts, dim=1)
            second_largest_indices = feature_topk_indices[:, self.k+neg_counts-1]  # 取每行的第二个最大值的索引

            # entity_cos_sim = self.cos(des_entity_vec.unsqueeze(1),des_entity_vec.unsqueeze(0))
            entity_cos_sim = self.cos(sen_entity_vec.unsqueeze(1),sen_entity_vec.unsqueeze(0))
            # print(entity_cos_sim[0])
            entity_topk_values, entity_topk_indices = torch.topk(entity_cos_sim, k=self.k+neg_counts, dim=1)
            # print(entity_topk_values[0])
            # print(entity_topk_indices[0])
            entity_second_largest_indices = entity_topk_indices[:,self.k+neg_counts-1]  # 取每行的第二个最大值的索引

            des_neg_fea_list = []

            for i, ind in enumerate(second_largest_indices):
                # per_des_neg_vec = torch.cat([des_vec_feature[ind],sen_entity_vec[i]],dim=0)
                des_neg_fea_list.append(des_vec_feature[ind])
            des_neg_fea_vec = torch.stack(des_neg_fea_list,dim=0)

            entity_neg_list = []
            for j, ind in enumerate(entity_second_largest_indices):
                entity_neg_list.append(sen_entity_vec[ind])
            entity_neg_vec = torch.stack(entity_neg_list,dim=0)
            
            des_neg_vec = torch.cat([des_neg_fea_vec,entity_neg_vec],dim=-1)
 
            # 双塔loss

            pos_cos_sim = self.cos(sen_vec,des_vec).unsqueeze(-1)
            neg_cos_sim = []
            for i in range(batch_size):
                # 其他不对应的 des_output 作为负样本
                neg_samples = [des_vec[j] for j in range(batch_size) if j != i]
                # 对应的 des_neg_output 作为负样本
                neg_samples.append(des_neg_vec[i])
                neg_samples = torch.stack(neg_samples, dim=0)
                neg_cos_sim_i = self.cos(sen_vec[i].unsqueeze(0), neg_samples)
                neg_cos_sim.append(neg_cos_sim_i)
        
            neg_cos_sim = torch.stack(neg_cos_sim, dim=0)
            cos_sim = torch.cat([pos_cos_sim, neg_cos_sim], dim=-1)
            #print(f'cos_sim:{cos_sim[0]}')
            con_labels = torch.zeros(batch_size).long().to(device)
            # con_labels_with_neg = torch.cat([con_labels,con_labels],dim=0)
            
            loss_fct = nn.CrossEntropyLoss()
            ori_loss = loss_fct(cos_sim  / 0.02, con_labels)

            loss = ori_loss 

            # detach: classify模型的梯度不会回传到EMMA，也就是两个模型互不影响
            input_ids, att_masks, token_type_ids, target_idx_arr, vec_idx_arr, sen_vec_, des_vec_ = \
                self.build_classifaction_input(sen_input_ids.detach(), des_input_ids.detach(), sen_output.detach(), des_output.detach(), training=True)

            # print(f'input_ids: {input_ids}')
            # print(f'att_masks: {att_masks}')
            # print(f'token_type_ids: {token_type_ids}')
            # print(f'target_idx_arr: {target_idx_arr}')
            # print(f'vec_idx_arr: {vec_idx_arr}')

            classify_loss = self.classify(input_ids.detach(), att_masks.detach(), token_type_ids.detach(), vec_idx_arr.detach(), sen_vec_.detach(), des_vec_.detach(), marked_e1.detach(), marked_e2.detach(), target_idx_arr.detach())
            # logger.info("loss: {}".format(loss))
            # logger.info("classify_loss: {}".format(classify_loss))
            
            return loss + classify_loss

        else:
            # cos_sim_ctx = self.cos(context_proj.unsqueeze(1), eval_des_features[:,0,:].unsqueeze(0)) # [32, 1, 768] # [1, m, 768]
            # cos_sim_e1 = self.cos(e1_proj.unsqueeze(1), eval_des_features[:,1,:].unsqueeze(0)) # expand : [32, m, 768] # [32, m, 768]
            # cos_sim_e2 = self.cos(e2_proj.unsqueeze(1), eval_des_features[:,2,:].unsqueeze(0)) # result : [32, m]
            # integrated_sim = self.alpha*(cos_sim_e1 + cos_sim_e2) + (1-self.alpha)*cos_sim_ctx
            # cos_sim = self.cos(sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0)) # [bs, 1, 768]   [1, m, 768] -> [bs, m]
            
            # sen_vec = self.get_sen_vec(sen_output, marked_e1, marked_e2)

            # sen_vec_feature = torch.mean(sen_output,dim=1)
            sen_vec_feature = torch.mean(sen_output,dim=1)
            # sen_vec_feature = self.sen_feature(torch.cat([sen_cls,sen_e1_entity_vec, sen_e2_entity_vec],dim=-1)) 
            sen_vec = torch.cat([sen_vec_feature,sen_entity_vec],dim=-1)
            # des_e1_weight_matirx = torch.einsum('bmi,ij,cqj->bcmq',self.des_vectors, self.des_e1, sen_output)

            # sen_e1_feature = self.e1_entity_mlp(torch.cat([sen_e1_entity_vec,sen_entity_between_vec],dim=-1))
            des_e1_layer = torch.squeeze(torch.add(torch.einsum('bmi,ij,cqj->bcmq',self.des_vectors, self.des_e1, sen_e1_entity_vec.unsqueeze(1)),self.des_bias1),dim=-1) #[bs_des,bs_sen,des_ml]
            des_bs_e1_weight, _ = torch.max(des_e1_layer, dim=1) #[bs_des,des_ml,des_ml]
            des_e1_layer1_softmax = torch.softmax(des_bs_e1_weight,dim=1) #[bs,ml]
            des_e1_vec = torch.sum(torch.unsqueeze(des_e1_layer1_softmax, dim=-1) * self.des_vectors, dim=1)

            # sen_e2_feature = self.e2_entity_mlp(torch.cat([sen_e2_entity_vec,sen_entity_between_vec],dim=-1))
            des_e2_layer = torch.squeeze(torch.add(torch.einsum('bmi,ij,cqj->bcmq',self.des_vectors, self.des_e2, sen_e2_entity_vec.unsqueeze(1)),self.des_bias2),dim=-1)
            des_bs_e2_weight, _ = torch.max(des_e2_layer, dim=1) #[bs_des,des_ml,des_ml]
            des_e2_layer1_softmax = torch.softmax(des_bs_e2_weight,dim=1) #[bs,ml]
            des_e2_vec = torch.sum(torch.unsqueeze(des_e2_layer1_softmax, dim=-1) * self.des_vectors, dim=1)

            des_en_layer = torch.squeeze(torch.add(torch.einsum('bmi,ij,cqj->bcmq',self.des_vectors, self.des_en, sen_entity_between_vec.unsqueeze(1)),self.des_bias3),dim=-1)
            des_bs_en_weight, _ = torch.max(des_en_layer, dim=1) #[bs_des,des_ml,des_ml]
            des_en_layer_softmax = torch.softmax(des_bs_en_weight,dim=1) #[bs,ml]
            des_en_vec = torch.sum(torch.unsqueeze(des_en_layer_softmax, dim=-1) * self.des_vectors, dim=1)


            # des_bs_e1_weight, _ = torch.max(des_e1_weight_matirx, dim=1) #[bs_des,des_ml,sen_ml]
            # des_bs_e2_weight, _ = torch.max(des_e2_weight_matirx, dim=1)

            # des_e1_weight, _ = torch.max(des_bs_e1_weight, dim=-1)
            # des_e2_weight, _ = torch.max(des_bs_e2_weight, dim=-1)

            # print(des_e1_weight_ind.shape)
            # print(des_e2_weight_ind.shape)

            #聚合步骤
            # des_e1_weight = torch.softmax(des_e1_weight,dim=-1)  #[bs_des,des_ml]
            # des_e2_weight = torch.softmax(des_e2_weight,dim=-1)

            # des_e1_vec = torch.sum(torch.unsqueeze(des_e1_weight,dim=-1)*self.des_vectors, dim=1)
            # des_e2_vec = torch.sum(torch.unsqueeze(des_e2_weight,dim=-1)*self.des_vectors, dim=1)
            # print("des_e1_weight:",des_e1_weight.shape)
            # print("des_e2_weight:",des_e2_weight.shape)
            
            # des_e1_output = torch.einsum('bci,bij->bcij', des_e1_weight, self.des_vectors)
            # des_e2_output = torch.einsum('bci,bij->bcij', des_e2_weight, self.des_vectors)            

            # des_e1_weighted_output, _ = torch.max(des_e1_output,dim=1) #[bs_sen,ml,hs]
            # des_e2_weighted_output, _ = torch.max(des_e2_output,dim=1)
            # des_feature_outputed, _ = torch.max(des_feature_output,dim=1)

            # des_feature_vec = self.des_feature_mlp(torch.cat([torch.mean(self.des_vectors,dim=1),des_en_vec],dim=-1))
            des_feature_vec = torch.mean(self.des_vectors,dim=1)
            # des_feature_vec = self.des_vectors[:,0,:]
            # des_e1_vec = torch.sum(des_e1_weighted_output, dim=1) 
            # des_e2_vec = torch.sum(des_e2_weighted_output, dim=1)

            des_entity_vec = F.normalize(torch.cat([des_e1_vec,des_e2_vec,des_en_vec],dim=1),p=2,dim=1) #[bs,2*hs]

            des_vec = torch.cat([des_feature_vec,des_entity_vec],dim=-1)
            # cos_sim = self.cos(sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0))
            cos_sim = self.cos(sen_vec.unsqueeze(1), des_vec.unsqueeze(0))
            max_sim, max_sim_idx = torch.max(cos_sim, dim=1)  # 获取相似度最大的一列
            
            # 分类
            top_k_values, top_k_indices = torch.topk(cos_sim, self.k, dim=1)
            # print(f'sen_vec:{sen_vec.shape}')
            input_ids, att_masks, token_type_ids, vec_idx_arr, sen_vec_, des_vec_ = \
                self.build_classifaction_input(sen_input_ids, self.des_input_ids_for_predict, sen_output, self.des_vectors, training=False, top_k_indices=top_k_indices)
            # print(f'input_ids: {input_ids.shape}')
            # print(f'att_masks: {att_masks.shape}')
            # print(f'token_type_ids: {token_type_ids.shape}')
            # print(f'vec_idx_arr: {vec_idx_arr.shape}')
            # print(f'sen_vec: {sen_vec.shape}')
            # print(f'des_vec: {des_vec.shape}')
            # print(f'sen_vec:{sen_vec.shape}')
            max_classify_idx = self.classify(input_ids, att_masks, token_type_ids, vec_idx_arr, sen_vec_, des_vec_, marked_e1, marked_e2)
            # print(f'max_sim_idx: {max_sim_idx}')
            # outputs = (outputs,) + max_sim_idx
            return max_sim_idx, max_classify_idx
        # return outputs
    
    
    def gen_des_vectors(self, des_features):
        # print(des_features.shape)
        max_len = des_features.shape[1]
        des_input_ids = des_features[:, 0: int(max_len / 2)]
        self.des_input_ids_for_predict = des_input_ids
        des_attention_masks = des_features[:, int(max_len / 2): max_len]

        des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_attention_masks,
            )
        des_output = des_outputs.last_hidden_state
        # self.des_vectors = self.get_des_vec(des_output)
        self.des_vectors = des_output

    def get_sen_vec(self, sen_output, marked_e1, marked_e2):
        if self.add_auto_match:
            en_h = self.get_sen_entity_between_vec(sen_output, marked_e1, marked_e2)
            e1_h = extract_entity(sen_output, marked_e1) # [E1] [bs, hs]
            e2_h = extract_entity(sen_output, marked_e2) # [E2]
            # sen_cls = sen_output[:, 0, :] # [bs, hs]
            sen_cls = torch.mean(sen_output,dim=1)
            # e1_h = torch.unsqueeze(e1_h, dim=1)
            # e2_h = torch.unsqueeze(e2_h, dim=1)
            # sen_cls = torch.unsqueeze(e1_h, dim=1) # [bs, 1, hs]
            # sen_vec = self.sen_att(e1_h, e2_h, sen_cls)
            # sen_vec = torch.squeeze(sen_vec, dim=1) # [bs, hs]
            # sen_feature_vec = self.sen_feature_mlp(torch.cat([sen_cls,en_h],dim=-1))
            sen_entity_vec = F.normalize(torch.cat([e1_h,e2_h,en_h],dim=1),p=2,dim=1)
            sen_vec = torch.cat([sen_cls, sen_entity_vec], dim=1) # [bs, 3* hs]
        else:
            sen_vec = sen_output[:, 0, :] # [cls]
        return sen_vec
    
    def get_sen_entity_between_vec(self, sen_output, marked_e1, marked_e2):

        e1_idx = torch.nonzero(marked_e1==1)
        e1_idx = torch.tensor([e1_idx[i][1] for i in range(len(e1_idx))])
        
        e2_idx = torch.nonzero(marked_e2==1)
        e2_idx = torch.tensor([e2_idx[i][1] for i in range(len(e2_idx))])     

        for i in range(len(e2_idx)):
            if e1_idx[i] < e2_idx[i]:
                marked_e1[i][e1_idx[i]:e2_idx[i]+1] = 1 
            else:
                marked_e1[i][e2_idx[i]:e1_idx[i]+1] = 1 

        extended_e_mask = marked_e1.unsqueeze(-1)
        # print(extended_e_mask.size()) # [32, 128, 1]
        extended_e_mask = extended_e_mask.float() * sen_output
        extended_e_mask = torch.mean(extended_e_mask,dim=-2)
        # print(extended_e_mask.size()) # [32, 768]
        # extended_e_mask = torch.stack([sequence_output[i,j,:] for i,j in enumerate(e_mask)])
        # print(extended_e_mask)
        return extended_e_mask.float()         



    def get_sen_entity_vec(self, sen_output, marked_e1, marked_e2):
        # if self.add_auto_match:

        e1_h = extract_entity(sen_output, marked_e1) # [E1] [bs, hs]
        e2_h = extract_entity(sen_output, marked_e2) # [E2]
            # sen_cls = sen_output[:, 0, :] # [bs, hs]
            # e1_h = torch.unsqueeze(e1_h, dim=1)
            # e2_h = torch.unsqueeze(e2_h, dim=1)
            # sen_cls = torch.unsqueeze(e1_h, dim=1) # [bs, 1, hs]
            # sen_vec = self.sen_att(e1_h, e2_h, sen_cls)
            # sen_vec = torch.squeeze(sen_vec, dim=1) # [bs, hs]
            # sen_vec = torch.cat([sen_cls, e1_h, e2_h], dim=1) # [bs, 3* hs]
        return e1_h,e2_h
    

    def get_des_vec(self, des_output):
        if self.add_auto_match:
            # [bs, ml, hs] x [hs, 1]
            des_cls = torch.mean(des_output,dim=1)
            # des_cls = des_output[:, 0, :]
            # des_output = des_output[:, 1:, :]
            # bert_layer1 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights1), self.des_bias1), dim=-1) #[bs, ml-1]
            # des_e1, _ = torch.max(torch.matmul(des_output, self.des_e1), dim=1)
            # des_e1_weight_matirx = torch.einsum('bmi,ij,aqj->bmq',des_output, self.des_e1, sen_output)
            des_e1 = torch.mean(torch.matmul(des_output, self.des_e1), dim=1)
            # bert_layer1 = torch.squeeze(des_e1, dim=-1) #[bs, ml-1]
            # bert_layer_softmax1 = torch.softmax(des_e1, dim=-1) # [bs, ml-1]
            # e1_h = torch.sum(torch.unsqueeze(bert_layer_softmax1, dim=2) * des_output, dim=1) # [bs, ml-1, 1] * [bs, ml-1, hs]
            # bert_layer2 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights2), self.des_bias2), dim=-1) #[bs, ml-1]
            # des_e2, _ = torch.max(torch.matmul(des_output, self.des_e2), dim=1)
            des_e2 = torch.mean(torch.matmul(des_output, self.des_e2), dim=1)
            # bert_layer2 = torch.squeeze(des_e2, dim=-1) #[bs, ml-1]
            # bert_layer_softmax2 = torch.softmax(des_e2, dim=-1) # [bs, ml - 1]
            # e2_h = torch.sum(torch.unsqueeze(bert_layer_softmax2, dim=2) * des_output, dim=1)
            des_entity_vec = F.normalize(torch.cat([des_e1,des_e2],dim=1),p=2,dim=1)
            des_vec = torch.cat([des_cls, des_entity_vec], dim=1) # [bs, 3* hs]
        else:
            des_vec = des_output[:, 0, :]
        return des_vec

    # def reparameterize(self, mu, logvar):
    #     sigma = torch.exp(0.5*logvar)
    #     eps = torch.FloatTensor(sigma.size()).normal_(
    #         0, 1).to(mu.device)
    #     return eps*sigma + mu

    # def KL_divergence(self, mu, logvar):
    #     return 0.5*(torch.sum(- (mu**2) + 1 + logvar - torch.exp(logvar)))/mu.shape[0]

    def build_classifaction_input(self, sen_input_ids, des_input_ids, sen_vec, des_vec, training=True, top_k_indices=None):
        '''
        sen_input_ids: [bs, max_seq_length]
        des_input_ids: [bs, max_seq_length](train) or [k, max_seq_length](predict)
        sen_vec: [bs, hs]
        des_vec: [bs, hs](train) or [k, hs](predict)
        training: for train or predict, True for train, False for predict
        '''
        # print(f'build_classifaction_input :: training: {training}')
        device = sen_input_ids.device
        k = self.k
        batch_size = len(sen_input_ids)
        # 选取负样本的策略：随机k个 or vec最接近的k个
        total_sen_tokens = []
        for input_ids in sen_input_ids:
            input_ids_trimmed = self.trim_post_zero(input_ids)
            sen_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_trimmed)
            total_sen_tokens.append(' '.join(sen_tokens))
        
        total_des_tokens = []
        for input_ids in des_input_ids:
            input_ids_trimmed = self.trim_post_zero(input_ids)
            des_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_trimmed)
            total_des_tokens.append(' '.join(des_tokens))

        input_ids = [] # [bs, k, ml]
        att_masks = [] # [bs, k, ml]
        token_type_ids = [] # [bs, k, ml]
        target_idx_arr = [] # [bs]
        vec_idx_arr = [] # [bs, k]
        # e1_mark = []
        # e2_mark = []

        for idx in range(batch_size):
            input_ids_per_sample = [] # [k, ml]
            att_masks_per_sample = [] # [k, ml]
            token_type_ids_per_sample = [] # [k, ml]
            sen_tokens = total_sen_tokens[idx]
            vec_idx_arr_per_sample = []
            # per_e1_mark = []
            # per_e2_mark = []

            if training:
                # 随机添加，保证分类模型类别均衡
                target_idx = random.randint(0, k - 1)
            
            current_idx = 0
            while(current_idx < k):
                if training:
                    if current_idx == target_idx:
                        pos_des_tokens = total_des_tokens[idx]
                        encode_info = self.tokenizer._encode_plus(sen_tokens, pos_des_tokens)
                        vec_idx_arr_per_sample.append(idx)
                        # per_e1_mark.append(marked_e1[idx])
                        # per_e2_mark.append(marked_e2[idx])
                    else:
                        # 随机选取负样本
                        neg_idx = random.randint(0, batch_size - 1)
                        # 跳过正样本本身
                        if neg_idx == idx:
                            continue
                        vec_idx_arr_per_sample.append(neg_idx)
                        neg_des_tokens = total_des_tokens[neg_idx]
                        encode_info = self.tokenizer._encode_plus(sen_tokens, neg_des_tokens)
                        # per_e1_mark.append(marked_e1[idx])
                        # per_e2_mark.append(marked_e2[idx])
                else:
                    vec_idx_arr_per_sample.append(top_k_indices[idx][current_idx])
                    des_tokens = total_des_tokens[top_k_indices[idx][current_idx]]
                    encode_info = self.tokenizer._encode_plus(sen_tokens, des_tokens)
                    # per_e1_mark.append(marked_e1[idx])
                    # per_e2_mark.append(marked_e2[idx])

                input_ids_per_sample.append(pad_or_truncate(torch.tensor(encode_info['input_ids']), self.max_seq_len).tolist())
                att_masks_per_sample.append(pad_or_truncate(torch.tensor(encode_info['attention_mask']), self.max_seq_len).tolist())
                token_type_ids_per_sample.append(pad_or_truncate(torch.tensor(encode_info['token_type_ids']), self.max_seq_len).tolist())
                current_idx += 1

            input_ids.append(input_ids_per_sample)
            att_masks.append(att_masks_per_sample)
            token_type_ids.append(token_type_ids_per_sample)
            vec_idx_arr.append(vec_idx_arr_per_sample)
            # e1_mark.append(per_e1_mark)
            # e2_mark.append(per_e2_mark)

            if training:
                target_idx_arr.append(target_idx)

        input_ids = torch.tensor(input_ids).to(device)
        att_masks = torch.tensor(att_masks).to(device)
        token_type_ids = torch.tensor(token_type_ids).to(device)
        vec_idx_arr = torch.tensor(vec_idx_arr).to(device)
        # e1_mark = [torch.stack(row) for row in e1_mark]
        # e2_mark = [torch.stack(row) for row in e2_mark]
        # print(f'e1_marked:{e1_mark}')
        # print(f'e1_marked:{e1_mark.shape}')
        # e1_mark = torch.stack(e1_mark).to(device)
        # e2_mark = torch.stack(e2_mark).to(device)
        if training:
            target_idx_arr = torch.tensor(target_idx_arr).to(device)

        if training:
            return input_ids, att_masks, token_type_ids, target_idx_arr, vec_idx_arr, sen_vec, des_vec
        else:
            return input_ids, att_masks, token_type_ids, vec_idx_arr, sen_vec, des_vec
    def trim_post_zero(self, vec):
        nonzero_indices = torch.nonzero(vec)
        num_nonzero = nonzero_indices.size(0)
        input_ids_trimmed = vec[:num_nonzero]
        return input_ids_trimmed


class Classify_model(nn.Module): 
    def __init__(self, pretrain_model_name_or_path, max_seq_len=128, k=3):
        super(Classify_model, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.bert = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.k = k

        # self.entity1_mlp =  nn.Linear(k * self.config.hidden_size, self.config.hidden_size)
        # self.entity2_mlp =  nn.Linear(k * self.config.hidden_size, self.config.hidden_size)

        self.mlp1 = nn.Linear(k*self.config.hidden_size, 512)
        self.mlp2 = nn.Linear(512,self.config.hidden_size)
        self.mlp3 = nn.Linear(self.config.hidden_size,k)

        # self.mha = MultiHeadAttention(self.config.hidden_size, 0.1, 8)

    def forward(self, input_ids, att_masks, token_type_ids, vec_idx_arr, sen_vec_arr, des_vec_arr, marked_e1, marked_e2, target_idx_arr=None):
        '''
        input_ids: [bs, k, ml]
        att_masks: [bs, k, ml]
        token_type_ids: [bs, k, ml]
        target_idx_arr: [bs]
        vec_idx_arr: [bs, k]
        sen_vec: [bs, hs]
        des_vec: [bs, hs]
        '''
        # print(vec_idx_arr)
        batch_size = len(input_ids)
        flatten_input_ids = input_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        flatten_att_masks = att_masks.view(-1, self.max_seq_len) # [bs * k, ml]
        flatten_token_type_ids = token_type_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        outputs = self.bert(
                input_ids=flatten_input_ids,
                attention_mask=flatten_att_masks,
                token_type_ids=flatten_token_type_ids
            )
        bert_output = outputs.last_hidden_state #[bs*k, ml, hs]
        # marked_e1 = marked_e1.view(-1,self.max_seq_len).bool().unsqueeze(-1)
        # marked_e2 = marked_e2.view(-1,self.max_seq_len).bool().unsqueeze(-1)
        # print(f'marked_e1:{torch.mean(bert_output.mask,dim=1).shape}')
        bert_output = bert_output[:, 0, :]  #[bs * k, hs]
        bert_output = bert_output.reshape(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]

        x = self.mlp1(bert_output) # [bs, k]
        x = torch.relu(x) # [bs, hs]
        x = self.mlp2(x) # [bs, hs]
        x = torch.relu(x)
        logits = self.mlp3(x)

        # x = torch.relu(x)
        # logits = self.mlp3(x)
        # print(f'logits: {logits.shape}')
        # print(f'target_idx_arr: {target_idx_arr}')
        if target_idx_arr != None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, target_idx_arr)
            # temp = nn.LogSoftmax(dim=-1)
            # temp_logits = temp(logits)
            # print(f'target_idx_arr: {target_idx_arr}')
            # print(f'logits: {temp_logits}')
            return loss
        else:
            max_idx = torch.argmax(logits, dim=1)
            # 由于max_idx（数量为k）和真实的idx（数量为unseen）不对应，所以需要转换
            converted_max_idx = vec_idx_arr[torch.arange(vec_idx_arr.size(0)), max_idx]
            # print(f'vec_idx_arr:{vec_idx_arr}')
            # print(f'max_idx: {max_idx}')
            # print(f'converted_max_idx: {converted_max_idx}')
            return converted_max_idx

# class MultiHeadAttention(nn.Module):

#     def __init__(self, hidden_size, dropout_rate, head_size=8) -> None:
#         super(MultiHeadAttention, self).__init__()
        
#         self.head_size = head_size
#         self.att_size = att_size = hidden_size // head_size
        
#         self.scale = att_size ** -0.5

#         self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)

#         self.att_dropout = nn.Dropout(dropout_rate)

#         self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias=False)

#     def forward(self, q, k, v, cache=None):
#         orig_q_size = q.size()

#         d_k = self.att_size
#         d_v = self.att_size
#         batch_size = q.size(0)

#         q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
#         k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
#         v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

#         q = q.transpose(1,2)
#         v = v.transpose(1,2)
#         k = k.transpose(1,2).transpose(2,3)

#         q.mul_(self.scale)
#         x = torch.matmul(q, k)
#         # x.masked_fill_(mask.unsqueeze(1), -1e9)

#         x = torch.softmax(x, dim=-1)
#         x = self.att_dropout(x)
#         x = x.matmul(v)

#         x = x.transpose(1,2).contiguous()
#         x = x.view(batch_size, -1, self.head_size*d_v)

#         x = self.output_layer(x)

#         assert x.size() == orig_q_size
#         return x
