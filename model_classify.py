import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForPreTraining, AlbertForPreTraining, DebertaV2PreTrainedModel, DebertaV2ForMaskedLM
from dataset import pad_or_truncate


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
        
        if add_auto_match:
            self.des_weights1 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
            self.des_weights2 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
        
        self.classify = Classify_model(pretrain_model_name_or_path, max_seq_len, k)
        

        self.des_vectors = None
    
    
    # def forward(self, sen_input_ids, sen_att_masks, des_input_ids, des_att_masks, sen_e1_pos, sen_e2_pos, sen_e1_pos_end, sen_e2_pos_end):
    def forward(self, sen_input_ids, sen_att_masks, des_input_ids=None, des_att_masks=None, \
                marked_e1=None, marked_e2=None, mark_head=None, mark_tail=None, head_range=None, tail_range=None):
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
        sen_vec = self.get_sen_vec(sen_output, marked_e1, marked_e2)


        if des_input_ids != None:
            des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_att_masks,
            )
            des_output = des_outputs.last_hidden_state
            des_vec = self.get_des_vec(des_output)
            
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

            # 双塔loss
            cos_sim = self.cos(sen_vec.unsqueeze(1), des_vec.unsqueeze(0)) # [bs, 1, 768]   [1, bs, 768]
            # print(cos_sim.shape)  #10 x 10
            con_labels = torch.arange(batch_size).long().to(device)
            loss_fct = nn.CrossEntropyLoss()            
            loss = loss_fct(cos_sim / 0.02, con_labels)

            # 分类loss
            # print(f'sen_input_ids: {sen_input_ids}')
            # print(f'des_input_ids: {des_input_ids}')
            # print(f'sen_vec: {sen_vec}')
            # print(f'des_vec: {des_vec}')

            # detach: classify模型的梯度不会回传到EMMA，也就是两个模型互不影响
            st_input_ids, st_att_masks, ds_input_ids, ds_att_masks, target_idx_arr, vec_idx_arr, sen_vec_, des_vec_ = \
                self.build_classifaction_input(sen_input_ids.detach(), des_input_ids.detach(), sen_vec.detach(), des_vec.detach(), training=True)
                
            # print('st_input_ids:', st_input_ids.shape)
            # print('st_att_masks:', st_att_masks.shape)
            # print('ds_input_ids:', ds_input_ids.shape)
            # print('ds_att_masks:', ds_att_masks.shape)
            # print(f'input_ids: {input_ids}')
            # print(f'att_masks: {att_masks}')
            # print(f'token_type_ids: {token_type_ids}')
            # print(f'target_idx_arr: {target_idx_arr}')
            # print(f'vec_idx_arr: {vec_idx_arr}')
            
            
            classify_loss = self.classify(st_input_ids.detach(), st_att_masks.detach(), ds_input_ids.detach(), ds_att_masks.detach(), vec_idx_arr.detach(), sen_vec_.detach(), des_vec_.detach(), target_idx_arr.detach())

            return loss + classify_loss
        else:
            # cos_sim_ctx = self.cos(context_proj.unsqueeze(1), eval_des_features[:,0,:].unsqueeze(0)) # [32, 1, 768] # [1, m, 768]
            # cos_sim_e1 = self.cos(e1_proj.unsqueeze(1), eval_des_features[:,1,:].unsqueeze(0)) # expand : [32, m, 768] # [32, m, 768]
            # cos_sim_e2 = self.cos(e2_proj.unsqueeze(1), eval_des_features[:,2,:].unsqueeze(0)) # result : [32, m]
            # integrated_sim = self.alpha*(cos_sim_e1 + cos_sim_e2) + (1-self.alpha)*cos_sim_ctx
            # cos_sim = self.cos(sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0)) # [bs, 1, 768]   [1, m, 768] -> [bs, m]
            cos_sim = self.cos(sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0))
            max_sim, max_sim_idx = torch.max(cos_sim, dim=1)  # 获取相似度最大的一列
            
            # 分类
            top_k_values, top_k_indices = torch.topk(cos_sim, self.k, dim=1)

            
            # print('sen_input_ids',sen_input_ids)
            # print('self.des_input_ids_for_predict',len(self.des_input_ids_for_predict))
            
            st_input_ids, st_att_masks, ds_input_ids, ds_att_masks, vec_idx_arr, sen_vec, des_vec = \
                self.build_classifaction_input(sen_input_ids, self.des_input_ids_for_predict, sen_vec, self.des_vectors, training=False, top_k_indices=top_k_indices)
            # print(f'input_ids: {input_ids.shape}')
            # print(f'att_masks: {att_masks.shape}')
            # print(f'token_type_ids: {token_type_ids.shape}')
            # print(f'vec_idx_arr: {vec_idx_arr.shape}')
            # print(f'sen_vec: {sen_vec.shape}')
            # print(f'des_vec: {des_vec.shape}')
            
            # print('ds_input_ids:', ds_input_ids.shape)
            # max_classify_idx = self.classify(st_input_ids, st_att_masks, st_token_type_ids, ds_input_ids, ds_att_masks, ds_token_type_ids, vec_idx_arr, sen_vec, des_vec)
            max_classify_idx = self.classify(st_input_ids, st_att_masks, ds_input_ids, ds_att_masks, vec_idx_arr, sen_vec, des_vec)
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
        self.des_vectors = self.get_des_vec(des_output)

    def get_sen_vec(self, sen_output, marked_e1, marked_e2):
        if self.add_auto_match:
            e1_h = extract_entity(sen_output, marked_e1) # [E1] [bs, hs]
            e2_h = extract_entity(sen_output, marked_e2) # [E2]
            sen_cls = sen_output[:, 0, :] # [bs, hs]
            # e1_h = torch.unsqueeze(e1_h, dim=1)
            # e2_h = torch.unsqueeze(e2_h, dim=1)
            # sen_cls = torch.unsqueeze(e1_h, dim=1) # [bs, 1, hs]
            # sen_vec = self.sen_att(e1_h, e2_h, sen_cls)
            # sen_vec = torch.squeeze(sen_vec, dim=1) # [bs, hs]
            sen_vec = torch.cat([sen_cls, e1_h, e2_h], dim=1) # [bs, 3* hs]
        else:
            sen_vec = sen_output[:, 0, :] # [cls]
        return sen_vec
    

    def get_des_vec(self, des_output):
        if self.add_auto_match:
            # [bs, ml, hs] x [hs, 1]
            des_cls = des_output[:, 0, :]
            des_output = des_output[:, 1:, :]
            bert_layer1 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights1), self.des_bias1), dim=-1) #[bs, ml-1]
            bert_layer_softmax1 = torch.softmax(bert_layer1, dim=-1) # [bs, ml-1]
            e1_h = torch.sum(torch.unsqueeze(bert_layer_softmax1, dim=2) * des_output, dim=1) # [bs, ml-1, 1] * [bs, ml-1, hs]
            bert_layer2 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights2), self.des_bias2), dim=-1) #[bs, ml-1]
            bert_layer_softmax2 = torch.softmax(bert_layer2, dim=-1) # [bs, ml - 1]
            e2_h = torch.sum(torch.unsqueeze(bert_layer_softmax2, dim=2) * des_output, dim=1)
            des_vec = torch.cat([des_cls, e1_h, e2_h], dim=1) # [bs, 3* hs]
        else:
            des_vec = des_output[:, 0, :]
        return des_vec

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
        total_st_input_ids = []
        for input_ids in sen_input_ids:
            # input_ids_trimmed = self.trim_post_zero(input_ids)
            sen_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            # print(f'sen_tokens: {sen_tokens}')
            total_st_input_ids.append(' '.join(sen_tokens))
        
        
        total_ds_input_ids = []
        for input_ids in des_input_ids:
            # input_ids_trimmed = self.trim_post_zero(input_ids)
            des_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            total_ds_input_ids.append(' '.join(des_tokens))

        # print(len(total_ds_input_ids))
        # print(len(total_st_input_ids))
        
        st_input_ids = [] # [bs, k, ml]
        st_att_masks = [] # [bs, k, ml]
        # st_token_type_ids = [] # [bs, k, ml]
        
        # print('ds_input_ids:',des_input_ids.shape)
        
        ds_input_ids = [] # [bs, k, ml]
        ds_att_masks = [] # [bs, k, ml]
        # ds_token_type_ids = [] # [bs, k, ml]
        
        target_idx_arr = [] # [bs]
        vec_idx_arr = [] # [bs, k]
        for idx in range(batch_size):
            sen_input_ids_per_sample = [] # [k, ml]
            sen_att_masks_per_sample = [] # [k, ml]
            # sen_token_type_ids_per_sample = [] # [k, ml]
            
            
            des_input_ids_per_sample = [] # [k, ml]
            des_att_masks_per_sample = [] # [k, ml]
            # des_token_type_ids_per_sample = [] # [k, ml]
            
            sen_tokens = total_st_input_ids[idx]
            # des_tokens = total_ds_input_ids[idx]
            vec_idx_arr_per_sample = []

            if training:
                # 随机添加，保证分类模型类别均衡
                target_idx = random.randint(0, k - 1)
            
            current_idx = 0
            while(current_idx < k):
                if training:
                    if current_idx == target_idx:
                        pos_des_tokens = total_ds_input_ids[idx]
                        sen_encode_info = self.tokenizer._encode_plus(sen_tokens)
                        des_encode_info = self.tokenizer._encode_plus(pos_des_tokens)
                        vec_idx_arr_per_sample.append(idx)
                    else:
                        # 随机选取负样本
                        neg_idx = random.randint(0, batch_size - 1)
                        # 跳过正样本本身
                        if neg_idx == idx:
                            continue
                        vec_idx_arr_per_sample.append(neg_idx)
                        neg_des_tokens = total_ds_input_ids[neg_idx]
                        sen_encode_info = self.tokenizer._encode_plus(sen_tokens)
                        des_encode_info = self.tokenizer._encode_plus(neg_des_tokens)
                else:
                    vec_idx_arr_per_sample.append(top_k_indices[idx][current_idx])
                    des_tokens = total_ds_input_ids[top_k_indices[idx][current_idx]]
                    sen_encode_info = self.tokenizer._encode_plus(sen_tokens)
                    des_encode_info = self.tokenizer._encode_plus(des_tokens)                    

                
                #sen
                sen_input_ids_per_sample.append(pad_or_truncate(torch.tensor(sen_encode_info['input_ids']), self.max_seq_len).tolist())
                sen_att_masks_per_sample.append(pad_or_truncate(torch.tensor(sen_encode_info['attention_mask']), self.max_seq_len).tolist())
                # sen_token_type_ids_per_sample.append(pad_or_truncate(torch.tensor(sen_encode_info['token_type_ids']), self.max_seq_len).tolist())
                
                
                #des
                des_input_ids_per_sample.append(pad_or_truncate(torch.tensor(des_encode_info['input_ids']), self.max_seq_len).tolist())
                des_att_masks_per_sample.append(pad_or_truncate(torch.tensor(des_encode_info['attention_mask']), self.max_seq_len).tolist())
                # des_token_type_ids_per_sample.append(pad_or_truncate(torch.tensor(des_encode_info['token_type_ids']), self.max_seq_len).tolist())
                
                current_idx += 1

            st_input_ids.append(sen_input_ids_per_sample)
            st_att_masks.append(sen_att_masks_per_sample)
            # st_token_type_ids.append(sen_token_type_ids_per_sample)
            
            ds_input_ids.append(des_input_ids_per_sample)
            ds_att_masks.append(des_att_masks_per_sample)
            # st_token_type_ids.append(des_token_type_ids_per_sample)
            
            vec_idx_arr.append(vec_idx_arr_per_sample)
            if training:
                target_idx_arr.append(target_idx)

        sent_input_ids = torch.tensor(st_input_ids).to(device)
        sent_att_masks = torch.tensor(st_att_masks).to(device)
        # sent_token_type_ids = torch.tensor(st_token_type_ids).to(device)
        
        desp_input_ids = torch.tensor(ds_input_ids).to(device)
        desp_att_masks = torch.tensor(ds_att_masks).to(device)
        # desp_token_type_ids = torch.tensor(ds_token_type_ids).to(device)
        
        # print('ds_input_ids:',des_input_ids.shape)
        # print(1111)
        
        vec_idx_arr = torch.tensor(vec_idx_arr).to(device)
        if training:
            target_idx_arr = torch.tensor(target_idx_arr).to(device)

        if training:
            # print('sent_input_ids:',sent_input_ids.shape)
            # print('ds_input_ids:',des_input_ids.shape)
            # print(1)
            return sent_input_ids, sent_att_masks, desp_input_ids, desp_att_masks, target_idx_arr, vec_idx_arr, sen_vec, des_vec
        else:
            # print('ds_input_ids:',des_input_ids.shape)
            # print(2)
            return sent_input_ids, sent_att_masks, desp_input_ids, desp_att_masks, vec_idx_arr, sen_vec, des_vec

    # def trim_post_zero(self, vec):
    #     nonzero_indices = torch.nonzero(vec)
    #     num_nonzero = nonzero_indices.size(0)
    #     input_ids_trimmed = vec[:num_nonzero]
    #     return input_ids_trimmed


class Classify_model(nn.Module): 
    def __init__(self, pretrain_model_name_or_path, max_seq_len=128, k=3):
        super(Classify_model, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.bert = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.k = k
        
        self.mlp_v = nn.Parameter(torch.Tensor(self.k, self.config.hidden_size, self.config.hidden_size))
        self.mlp_v1 = nn.Linear(3 * self.config.hidden_size, self.config.hidden_size)
        
        self.mlp1 = nn.Linear(3*self.max_seq_len * self.config.hidden_size, self.config.hidden_size)
        self.mlp2 = nn.Linear(self.k*self.config.hidden_size, self.config.hidden_size)
        self.mlp3 = nn.Linear(self.config.hidden_size, self.k)
        
        
        self.mlp_v.data.normal_(mean=0.0, std=0.02)
        # self.mha = MultiHeadAttention(self.config.hidden_size, 0.1, 8)

    def forward(self, st_input_ids, st_att_masks,ds_input_ids, ds_att_masks, vec_idx_arr, sen_vec_arr, des_vec_arr, target_idx_arr=None):
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
        batch_size = len(st_input_ids)
        device = st_input_ids.device
        # st_input_ids = st_input_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        # st_att_masks = st_att_masks.view(-1, self.max_seq_len) # [bs * k, ml]
        # st_token_type_ids = st_token_type_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        
        # ds_input_ids = ds_input_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        # ds_att_masks = ds_att_masks.view(-1, self.max_seq_len) # [bs * k, ml]
        # ds_token_type_ids = ds_token_type_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        
        # print(type(ds_input_ids))
        
        # print(st_input_ids.shape)
        # print(ds_input_ids.shape)
        des_token_feature = [] #[bs,k,des_ml,hs]
        # des_feature_weight = [] #[bs,k,des_ml,hs]
        classify_input = []
        
        feature_cos = []
        
        for bs_idx in range(batch_size):
            
            
            per_st_input_ids = st_input_ids[bs_idx]
            per_st_att_masks = st_att_masks[bs_idx]
 
            per_ds_input_ids = ds_input_ids[bs_idx]
            per_ds_att_masks = ds_att_masks[bs_idx]            
            
            
            # print('per_sen_vec',per_st_input_ids.shape)
            # print('per_des_vec',per_ds_input_ids.shape)
            
            # for k in range(per_st_input_ids.shape(0)):
            sen_outputs = self.bert(
                input_ids=per_st_input_ids,
                attention_mask=per_st_att_masks,
            )
        
            des_outputs = self.bert(
                input_ids=per_ds_input_ids,
                attention_mask=per_ds_att_masks,
            )
            
            
            sen_bert_output = sen_outputs.last_hidden_state #[k,ml,hs]
            des_bert_output = des_outputs.last_hidden_state
            
            _, sen_ml, feature_dim = sen_bert_output.shape
            _, des_ml, _ = des_bert_output.shape  
            
            # print('per_sen_vec',sen_bert_output.shape)
            # print('per_des_vec',des_bert_output.shape)
             
            # per_sen_vec = sen_bert_output[bs_idx] #[k,sen_ml, feature_dim]
            # per_des_vec = des_bert_output[bs_idx]
            

            
            token_features = torch.einsum('bmn,bnk,bqk->bmq', des_bert_output, self.mlp_v, sen_bert_output)
            # print('token_features',token_features)
            combined = torch.cat((sen_bert_output.unsqueeze(1).expand(-1,des_ml,-1,-1),
                                  des_bert_output.unsqueeze(2).expand(-1,-1,sen_ml,-1),
                                  token_features.unsqueeze(-1).expand(-1,-1,-1,feature_dim)), dim=-1) #[k,des_ml,sen_ml, 3*feature_dim]
            # print('combined',combined)
            relation_feature = self.mlp_v1(combined) #[k,des_ml,sen_ml,feature_dim]
            # print('relation_feature',relation_feature)
            similar_score = torch.max(relation_feature,dim=2).values #[k,des_ml,feature_dim]
            # print('similar_score',(torch.exp(-similar_score)/0.02).shape)
            # print('similar_score',(torch.sum(torch.exp(-similar_score)/0.02, dim=1, keepdim=True)).shape)
            # print(((torch.exp(-similar_score)/0.02)/torch.sum(torch.exp(-similar_score)/0.02, dim=1, keepdim=True)).shape)
            des_feature_weight=(torch.exp(-similar_score)/0.02)/torch.sum(torch.exp(-similar_score)/0.02, dim=1, keepdim=True)
            des_similarity_embedding = torch.mul(des_bert_output,des_feature_weight)
            feature_cos.append(torch.mul(des_feature_weight,similar_score))
            # print('des_similarity_embedding',des_similarity_embedding.shape)
        # bert_output = bert_output[:, 0, :] # [bs * k, hs]
        # bert_output = torch.unsqueeze(bert_output, dim=1) # [bs * k, 1, hs]
        # bert_output = bert_output.view(batch_size, self.k, self.config.hidden_size) # [bs, k, hs]
        # bert_output = bert_output.reshape(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]
        # 根据vec_idx_arr取出对应的vec
        # flatten_idx_arr = vec_idx_arr.view(-1) # [bs * k]
        # sen_vec_s = torch.index_select(sen_vec_arr, 0, flatten_idx_arr) # [bs * k, hs]
        # des_vec_s = torch.index_select(des_vec_arr, 0, flatten_idx_arr) # [bs * k, hs]
        
        # # # sen_vec_s = torch.unsqueeze(sen_vec_s, dim=1) # [bs * k, 1, hs]
        # # # des_vec_s = torch.unsqueeze(des_vec_s, dim=1) # [bs * k, 1, hs]

        # sen_vec_s = sen_vec_s.view(batch_size, self.k, self.config.hidden_size) # [bs, k, hs]
        # des_vec_s = des_vec_s.view(batch_size, self.k, self.config.hidden_size) # [bs, k, hs]

        # # BERT output 和 DSSM vec交互方式
        # sen_att_output = self.mha(sen_vec_s, bert_output, bert_output) # [bs, k, hs]
        # des_att_output = self.mha(des_vec_s, bert_output, bert_output) # [bs, k, hs]

        # sen_att_output = sen_att_output.view(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]
        # des_att_output = des_att_output.view(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]

        # concatenated_output = torch.cat([sen_att_output, des_att_output], dim=1) # [bs, 2k * hs]
        
        # print(len(des_feature_weight))
        # print(torch.tensor(des_feature_weight).shape)
        
            # print((sen_bert_output.reshape(self.k, self.max_seq_len*self.config.hidden_size)))
            # print((sen_bert_output.reshape(self.k, self.max_seq_len*self.config.hidden_size)).shape)

            # print((des_similarity_embedding.reshape(self.k, self.max_seq_len*self.config.hidden_size)))
            # print((des_similarity_embedding.reshape(self.k, self.max_seq_len*self.config.hidden_size)).shape)


            # classify_input.append(torch.cat((sen_bert_output.reshape(self.k, self.max_seq_len*self.config.hidden_size),
            #                         des_similarity_embedding.reshape(self.k, self.max_seq_len*self.config.hidden_size),
            #                         des_bert_output.reshape(self.k, self.max_seq_len*self.config.hidden_size)),
            #                         dim =-1))
            
            classify_input.append(torch.cat((sen_bert_output.reshape(self.k, self.max_seq_len*self.config.hidden_size),
                                    des_similarity_embedding.reshape(self.k, self.max_seq_len*self.config.hidden_size)),
                                    dim =-1))
        
                   
        # print(cos_sim) 10 x 10
        feature_cos = torch.stack(feature_cos, dim=0)
        print(feature_cos.shape)
        con_labels = torch.arange(batch_size).long().to(device)
        print(con_labels)
        loss_fct = nn.CrossEntropyLoss()            
        info_loss = loss_fct(feature_cos / 0.02, con_labels)
        
        classify_input_embedding = torch.stack(classify_input, dim=0)
        # print(classify_input_embedding.shape)
        x = self.mlp1(classify_input_embedding) # [bs,k, hs]
        # x = self.mlp1(concatenated_output) # [bs, hs]
        x = torch.relu(x).reshape(batch_size,self.k*self.config.hidden_size) # [bs, hs]
        x = self.mlp2(x) # [bs, hs]
        x =torch.relu(x)
        logits =self.mlp3(x)
        
        # print(f'logits: {logits}')
        # print(f'target_idx_arr: {target_idx_arr}')
        if target_idx_arr != None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, target_idx_arr)
            # print(f'target_idx_arr: {target_idx_arr}')
            return loss+info_loss
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
