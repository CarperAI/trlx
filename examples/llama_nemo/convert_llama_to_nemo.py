import os

import torch
from transformers import AutoModelForCausalLM

# Constants
TOTAL_LAYERS = 32
TOTAL_TP = 1
HIDDEN_DIM = 4096
FFN_HIDDEN_DIM = 11008
PART_ATTN_DIM = HIDDEN_DIM // TOTAL_TP
PART_MLP_DIM = FFN_HIDDEN_DIM // TOTAL_TP
VOCAB_SIZE = 32000
EMBEDDING_DIM = VOCAB_SIZE // TOTAL_TP
INPUT_FOLDER = "llama-nemo-7b-tp4"  # NeMo initial checkpoint folder
OUTPUT_FOLDER = "llama-nemo-7b-converted-test"  # NeMo converted checkpoint folder with llama weights

# Model Loading
model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-7B-1.1-HF")
model_state_dict = model.state_dict()


def build_layer_mapping(layer):
    return {
        f"model.language_model.encoder.layers.{layer}.input_layernorm.weight": f"model.layers.{layer}.input_layernorm.weight",
        f"model.language_model.encoder.layers.{layer}.self_attention.query_key_value.weight": [
            f"model.layers.{layer}.self_attn.q_proj.weight",
            f"model.layers.{layer}.self_attn.k_proj.weight",
            f"model.layers.{layer}.self_attn.v_proj.weight",
        ],
        f"model.language_model.encoder.layers.{layer}.self_attention.dense.weight": f"model.layers.{layer}.self_attn.o_proj.weight",
        f"model.language_model.encoder.layers.{layer}.post_attention_layernorm.weight": f"model.layers.{layer}.post_attention_layernorm.weight",
        f"model.language_model.encoder.layers.{layer}.mlp.dense_h_to_4h.weight": f"model.layers.{layer}.mlp.gate_proj.weight",
        f"model.language_model.encoder.layers.{layer}.mlp.dense_h_to_4h_2.weight": f"model.layers.{layer}.mlp.up_proj.weight",
        f"model.language_model.encoder.layers.{layer}.mlp.dense_4h_to_h.weight": f"model.layers.{layer}.mlp.down_proj.weight",
    }


def load_nemo_state_dict(tp_idx):
    if TOTAL_TP == 1:
        return torch.load(f"{INPUT_FOLDER}/model_weights.ckpt")
    return torch.load(f"{INPUT_FOLDER}/mp_rank_0{tp_idx}/model_weights.ckpt")


def save_nemo_state_dict(nemo_state_dict, tp_idx):
    if TOTAL_TP == 1:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        torch.save(nemo_state_dict, f"{OUTPUT_FOLDER}/model_weights.ckpt")
    else:
        os.makedirs(f"{OUTPUT_FOLDER}/mp_rank_0{tp_idx}", exist_ok=True)
        torch.save(nemo_state_dict, f"{OUTPUT_FOLDER}/mp_rank_0{tp_idx}/model_weights.ckpt")


def map_weights(tp_idx):
    nemo_state_dict = load_nemo_state_dict(tp_idx)
    # Word embeddings mapping
    original_size = nemo_state_dict["model.language_model.embedding.word_embeddings.weight"].shape
    nemo_state_dict["model.language_model.embedding.word_embeddings.weight"] = model_state_dict[
        "model.embed_tokens.weight"
    ][tp_idx * EMBEDDING_DIM : (tp_idx + 1) * EMBEDDING_DIM, :]
    print(
        f"Embedding - model.language_model.embedding.word_embeddings.weight - {original_size} -> {nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape}"
    )
    assert nemo_state_dict["model.language_model.embedding.word_embeddings.weight"].shape == original_size
    # Final layer norm mapping
    original_size = nemo_state_dict["model.language_model.encoder.final_layernorm.weight"].shape
    nemo_state_dict["model.language_model.encoder.final_layernorm.weight"] = model_state_dict["model.norm.weight"]
    print(
        f"Final Layer Norm - model.language_model.encoder.final_layernorm.weight - {original_size} -> {nemo_state_dict['model.language_model.encoder.final_layernorm.weight'].shape}"
    )
    assert nemo_state_dict["model.language_model.encoder.final_layernorm.weight"].shape == original_size
    # Output layer weight mapping
    original_size = nemo_state_dict["model.language_model.output_layer.weight"].shape
    nemo_state_dict["model.language_model.output_layer.weight"] = model_state_dict["lm_head.weight"][
        tp_idx * EMBEDDING_DIM : (tp_idx + 1) * EMBEDDING_DIM, :
    ]
    print(
        f"Output Layer - model.language_model.output_layer.weight - {original_size} -> {nemo_state_dict['model.language_model.output_layer.weight'].shape}"
    )
    assert nemo_state_dict["model.language_model.output_layer.weight"].shape == original_size
    # Other layer mappings
    for layer in range(TOTAL_LAYERS):
        layer_mapping = build_layer_mapping(layer)
        for k in layer_mapping.keys():
            original_size = nemo_state_dict[k].shape
            if "self_attention.query_key_value.weight" in k:
                nemo_state_dict[k] = get_self_attention_weight(model_state_dict, layer_mapping, k, tp_idx)
            elif "self_attention.dense.weight" in k:
                nemo_state_dict[k] = model_state_dict[layer_mapping[k]][
                    :, tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM
                ]
            elif "mlp.dense_h_to_4h.weight" in k or "mlp.dense_h_to_4h_2.weight" in k:
                nemo_state_dict[k] = get_mlp_weight(model_state_dict, layer_mapping, k, tp_idx)
            elif "mlp.dense_4h_to_h.weight" in k:
                nemo_state_dict[k] = model_state_dict[layer_mapping[k]][
                    :, tp_idx * PART_MLP_DIM : (tp_idx + 1) * PART_MLP_DIM
                ]
            else:
                nemo_state_dict[k] = model_state_dict[layer_mapping[k]]
            print(f"Layer {layer} - {k} - {original_size} -> {nemo_state_dict[k].shape}")
            assert nemo_state_dict[k].shape == original_size
    save_nemo_state_dict(nemo_state_dict, tp_idx)


def get_self_attention_weight(model_state_dict, layer_mapping, key, tp_idx):
    llama_query = model_state_dict[layer_mapping[key][0]][tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM, :]
    llama_key = model_state_dict[layer_mapping[key][1]][tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM, :]
    llama_value = model_state_dict[layer_mapping[key][2]][tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM, :]
    return torch.cat([llama_query, llama_key, llama_value], dim=0)


def get_mlp_weight(model_state_dict, layer_mapping, key, tp_idx):
    llama_weight = model_state_dict[layer_mapping[key]]
    return llama_weight[tp_idx * PART_MLP_DIM : (tp_idx + 1) * PART_MLP_DIM, :]


for tp_idx in range(TOTAL_TP):
    map_weights(tp_idx)
