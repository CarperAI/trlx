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
total_tp = 4 
hidden_dim = 4096
ffn_hidden_dim = 11008
total_self_attn_dim = hidden_dim * 3
part_self_attn_dim = int(total_self_attn_dim / total_tp)
part_self_attn_dense_dim = int(hidden_dim / total_tp)
part_mlp_dim = int(ffn_hidden_dim / total_tp)
vocab_size = 32000
embedding_dim = int(vocab_size / total_tp)

def mapping(tp_idx):
    inp_folder = "llama-nemo-7b-tp4"
    if total_tp == 1:
        nemo_state_dict = torch.load(f"{inp_folder}/model_weights.ckpt")
    else:
        nemo_state_dict = torch.load(f"{inp_folder}/mp_rank_0{tp_idx}/model_weights.ckpt")
    original_size = nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape
    nemo_state_dict['model.language_model.embedding.word_embeddings.weight'] = llama_state_dict['model.embed_tokens.weight'][tp_idx * embedding_dim : (tp_idx + 1) * embedding_dim, :]
    print(f"Embedding - model.language_model.embedding.word_embeddings.weight - {original_size} -> {nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape}")
    assert nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape == original_size
    original_size = nemo_state_dict['model.language_model.encoder.final_layernorm.weight'].shape
    nemo_state_dict['model.language_model.encoder.final_layernorm.weight'] = llama_state_dict['model.norm.weight']
    print(f"Final Layer Norm - model.language_model.encoder.final_layernorm.weight - {original_size} -> {nemo_state_dict['model.language_model.encoder.final_layernorm.weight'].shape}")
    assert nemo_state_dict['model.language_model.encoder.final_layernorm.weight'].shape == original_size
    original_size = nemo_state_dict['model.language_model.output_layer.weight'].shape
    nemo_state_dict['model.language_model.output_layer.weight'] = llama_state_dict['lm_head.weight'][tp_idx * embedding_dim : (tp_idx + 1) * embedding_dim, :]
    print(f"Output Layer - model.language_model.output_layer.weight - {original_size} -> {nemo_state_dict['model.language_model.output_layer.weight'].shape}")
    assert nemo_state_dict['model.language_model.output_layer.weight'].shape == original_size
    for layer in range(total_layers):
        mapp = buil_layer_mapping(layer)
        for k in mapp.keys():
            original_size = nemo_state_dict[k].shape
            if "self_attention.query_key_value.weight" in k:
                part_self_attn_dim = hidden_dim // total_tp
                llama_query = llama_state_dict[mapp[k][0]][tp_idx * part_self_attn_dim: (tp_idx + 1) * part_self_attn_dim, :]
                llama_key = llama_state_dict[mapp[k][1]][tp_idx * part_self_attn_dim: (tp_idx + 1) * part_self_attn_dim, :]
                llama_value = llama_state_dict[mapp[k][2]][tp_idx * part_self_attn_dim: (tp_idx + 1) * part_self_attn_dim, :]
                nemo_state_dict[k] = torch.cat([llama_query, llama_key, llama_value], dim=0)
            elif "self_attention.dense.weight" in k:
                nemo_state_dict[k] = llama_state_dict[mapp[k]][:, tp_idx * part_self_attn_dense_dim : (tp_idx + 1) * part_self_attn_dense_dim]
            elif "mlp.dense_h_to_4h.weight" in k:
                llama_weight = llama_state_dict[mapp[k]]
                nemo_state_dict[k] = llama_weight[tp_idx * part_mlp_dim : (tp_idx + 1) * part_mlp_dim, :]
            elif "mlp.dense_h_to_4h_2.weight" in k:
                llama_weight = llama_state_dict[mapp[k]]
                nemo_state_dict[k] = llama_weight[tp_idx * part_mlp_dim : (tp_idx + 1) * part_mlp_dim , :]
            elif "mlp.dense_4h_to_h.weight" in k:
                llama_weight = llama_state_dict[mapp[k]]
                nemo_state_dict[k] = llama_weight[:, tp_idx *  part_mlp_dim : (tp_idx + 1) * part_mlp_dim]
            else:
                nemo_state_dict[k] = llama_state_dict[mapp[k]]
            print(f"Layer {layer} - {k} - {original_size} -> {nemo_state_dict[k].shape}")
            assert nemo_state_dict[k].shape == original_size
    return nemo_state_dict

for tp_idx in range(total_tp):
    nemo_state_dict = mapping(tp_idx)
    out_folder = "llama-nemo-7b-tp4-converted"
    if not os.path.exists(f"{out_folder}/mp_rank_0{tp_idx}"):
        os.makedirs(f"{out_folder}/mp_rank_0{tp_idx}")
    torch.save(nemo_state_dict, f"{out_folder}/mp_rank_0{tp_idx}/model_weights.ckpt")