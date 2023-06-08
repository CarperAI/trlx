# coding: utf-8
import os
import torch
from transformers import AutoModelForCausalLM

def buil_layer_mapping(layer):
    mapp = {
        f'model.language_model.encoder.layers.{layer}.input_layernorm.weight':  f'model.layers.{layer}.input_layernorm.weight',
        f'model.language_model.encoder.layers.{layer}.self_attention.query_key_value.weight': [f'model.layers.{layer}.self_attn.q_proj.weight', f'model.layers.{layer}.self_attn.k_proj.weight', f'model.layers.{layer}.self_attn.v_proj.weight'],
        f'model.language_model.encoder.layers.{layer}.self_attention.dense.weight': f'model.layers.{layer}.self_attn.o_proj.weight',
        f'model.language_model.encoder.layers.{layer}.post_attention_layernorm.weight': f'model.layers.{layer}.post_attention_layernorm.weight',
        f'model.language_model.encoder.layers.{layer}.mlp.dense_h_to_4h.weight': f'model.layers.{layer}.mlp.gate_proj.weight',
        f'model.language_model.encoder.layers.{layer}.mlp.dense_h_to_4h_2.weight': f'model.layers.{layer}.mlp.up_proj.weight',
        f'model.language_model.encoder.layers.{layer}.mlp.dense_4h_to_h.weight': f'model.layers.{layer}.mlp.down_proj.weight',
    }
    return mapp

llama_state_dict = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-7B-1.1-HF")
llama_state_dict = llama_state_dict.state_dict()
total_layers = 32
total_tp = 1
hidden_dim = 4096
ffn_hidden_dim = 11008
total_self_attn_dim = hidden_dim * 3
part_self_attn_dim = int(total_self_attn_dim / total_tp)
part_mlp_dim = int(ffn_hidden_dim / total_tp)
vocab_size = 32000
embedding_dim = int(vocab_size / total_tp)

def mapping(tp_idx):
    #nemo_state_dict = torch.load(f"llama-nemo-7b/mp_rank_0{tp_idx}/model_weights.ckpt")
    nemo_state_dict = torch.load(f"llama-nemo-7b/model_weights.ckpt")
    original_size = nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape
    nemo_state_dict['model.language_model.embedding.word_embeddings.weight'] = llama_state_dict['model.embed_tokens.weight'][tp_idx * embedding_dim : (tp_idx + 1) * embedding_dim, :]
    print(f"Embedding - model.language_model.embedding.word_embeddings.weight - {original_size} -> {nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape}")
    assert nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape == original_size
    original_size = nemo_state_dict['model.language_model.encoder.final_layernorm.weight'].shape
    nemo_state_dict['model.language_model.encoder.final_layernorm.weight'] = llama_state_dict['model.norm.weight']
    
    for layer in range(total_layers):
        mapp = buil_layer_mapping(layer)
        for k in mapp.keys():
            original_size = nemo_state_dict[k].shape
            if "self_attention.query_key_value.weight" in k:
                llama_attention_weight = torch.cat([llama_state_dict[mapp[k][0]], llama_state_dict[mapp[k][1]], llama_state_dict[mapp[k][2]]], dim=0)
                nemo_state_dict[k] = llama_attention_weight[tp_idx * part_self_attn_dim: (tp_idx + 1) * part_self_attn_dim]
            elif "mlp.dense_h_to_4h.weight" in k:
                llama_weight = llama_state_dict[mapp[k]]
                nemo_state_dict[k] = llama_weight[tp_idx * part_mlp_dim : (tp_idx + 1) * part_mlp_dim, :]
            elif "mlp.dense_h_to_4h_2.weight" in k:
                llama_weight = llama_state_dict[mapp[k]]
                nemo_state_dict[k] = llama_weight[:, tp_idx * part_mlp_dim : (tp_idx + 1) * part_mlp_dim]#.transpose(0, 1)
            elif "mlp.dense_4h_to_h.weight" in k:
                llama_weight = llama_state_dict[mapp[k]]
                nemo_state_dict[k] = llama_weight[tp_idx *  part_mlp_dim : (tp_idx + 1) * part_mlp_dim]#.transpose(0, 1)
            else:
                nemo_state_dict[k] = llama_state_dict[mapp[k]]
            print(f"Layer {layer} - {k} - {original_size} -> {nemo_state_dict[k].shape}")
            assert nemo_state_dict[k].shape == original_size
    return nemo_state_dict

for tp_idx in range(total_tp):
    nemo_state_dict = mapping(tp_idx)
    if not os.path.exists(f"llama-nemo-7b-converted/mp_rank_0{tp_idx}"):
        os.makedirs(f"llama-nemo-7b-converted/mp_rank_0{tp_idx}")
    torch.save(nemo_state_dict, f"llama-nemo-7b-converted/mp_rank_0{tp_idx}/model_weights.ckpt")
    

#state_dict = lst_nemo[0]
#layer = 10
#for w in state_dict.keys():
#    if f'layers.{layer}' in w:
#        print("Weight name: ", w, "Shape: ", state_dict[w].shape)
#
#def mapping_weight_name(num_layers):
#    llama_weight_names = ['model.embed_tokens.weight'] + sum([
#        [
#            f'model.layers.{layer}.self_attn.q_proj.weight', # 6656 x 6656
#            f'model.layers.{layer}.self_attn.k_proj.weight', # 6656 x 6656
#            f'model.layers.{layer}.self_attn.v_proj.weight', # 6656 x 6656
#            f'model.layers.{layer}.self_attn.o_proj.weight', # 6656 x 6656 
#            f'model.layers.{layer}.self_attn.rotary_emb.inv_freq', # 64
#            f'model.layers.{layer}.mlp.gate_proj.weight', # 17920 x 6656 
#            f'model.layers.{layer}.mlp.down_proj.weight', # 6656 x 17920
#            f'model.layers.{layer}.mlp.up_proj.weight', # 17920 x 6656
#            f'model.layers.{layer}.input_layernorm.weight', # 6656 
#            f'model.layers.{layer}.post_attention_layernorm.weight' # 6656
#        ]
#        for layer in range(num_layers)
#    ], [])
#    
#    nemo_weight_names = ['model.language_model.embedding.word_embeddings.weight', 'model.language_model.embedding.position_embeddings.weight'] # + sum([
#    lst2 = sum([[
#                f'model.language_model.encoder.layers.{layer}.input_layernorm.weight', # 6656
#                f'model.language_model.encoder.layers.{layer}.input_layernorm.bias', # 6656
#                f'model.language_model.encoder.layers.{layer}.self_attention.query_key_value.weight',  # 9984 x 6656
#                f'model.language_model.encoder.layers.{layer}.self_attention.dense.weight', # 6656 x 3328
#                f'model.language_model.encoder.layers.{layer}.post_attention_layernorm.weight', # 6656
#                f'model.language_model.encoder.layers.{layer}.post_attention_layernorm.bias', # 6656
#                f'model.language_model.encoder.layers.{layer}.mlp.dense_h_to_4h.weight', # 8960 x 6656 --> gate
#                f'model.language_model.encoder.layers.{layer}.mlp.dense_h_to_4h_2.weight', # 8960 x 6656 --> down
#                f'model.language_model.encoder.layers.{layer}.mlp.dense_4h_to_h.weight' # 6656 x 8960 --> up
#            ]
#                for layer in range(num_layers)], [])
#    return llama_weight_names, nemo_weight_names
#
