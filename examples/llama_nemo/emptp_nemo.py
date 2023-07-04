import os
from pathlib import Path

import torch
from omegaconf.omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

import trlx
from trlx.data.default_configs import default_ppo_config
from trlx.trainer.nemo_ppo_trainer import PPOGPT, megatron_trainer


def main(args):  # noqa: C901
    print("loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    toks = tokenizer("heyy", return_tensors="pt")
    print(toks)
    config = AutoConfig.from_pretrained(args.model_path)
    print("loaded model")


    # model_state_dict = model.state_dict()
    print("model loaded")
    # Constants
    TOTAL_LAYERS = config.num_hidden_layers
    TOTAL_TP = args.total_tp
    HIDDEN_DIM = config.hidden_size
    FFN_HIDDEN_DIM = config.intermediate_size
    PART_ATTN_DIM = HIDDEN_DIM // TOTAL_TP
    PART_MLP_DIM = FFN_HIDDEN_DIM // TOTAL_TP
    VOCAB_SIZE = config.vocab_size
    EMBEDDING_DIM = VOCAB_SIZE // TOTAL_TP
    # INPUT_FOLDER = "llama-nemo-65b-tp4-in"  # NeMo initial checkpoint folder
    OUTPUT_FOLDER = args.output_folder  # NeMo converted checkpoint folder with llama weights

    # Model Loading

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

    """
    def load_nemo_state_dict(tp_idx):
        if TOTAL_TP == 1:
            return torch.load(f"{INPUT_FOLDER}/model_weights.ckpt")
        return torch.load(f"{INPUT_FOLDER}/mp_rank_0{tp_idx}/model_weights.ckpt")
    """

    def save_nemo_state_dict(nemo_state_dict, tp_idx):
        if TOTAL_TP == 1:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            torch.save(nemo_state_dict, f"{OUTPUT_FOLDER}/model_weights.ckpt")
        else:
            os.makedirs(f"{OUTPUT_FOLDER}/mp_rank_0{tp_idx}", exist_ok=True)
            torch.save(nemo_state_dict, f"{OUTPUT_FOLDER}/mp_rank_0{tp_idx}/model_weights.ckpt")

    def map_weights(tp_idx):
        nemo_state_dict = {}
        # nemo_state_dict = load_nemo_state_dict(tp_idx)
        # Word embeddings mapping
        # original_size = nemo_state_dict["model.language_model.embedding.word_embeddings.weight"].shape
        nemo_state_dict["model.language_model.embedding.word_embeddings.weight"] = model_state_dict[
            "model.embed_tokens.weight"
        ][tp_idx * EMBEDDING_DIM : (tp_idx + 1) * EMBEDDING_DIM, :]
        # print(
        #    f"Embedding - model.language_model.embedding.word_embeddings.weight - {original_size} -> {nemo_state_dict['model.language_model.embedding.word_embeddings.weight'].shape}"
        # )
        # assert nemo_state_dict["model.language_model.embedding.word_embeddings.weight"].shape == original_size
        # Final layer norm mapping
        # original_size = nemo_state_dict["model.language_model.encoder.final_layernorm.weight"].shape
        nemo_state_dict["model.language_model.encoder.final_layernorm.weight"] = model_state_dict["model.norm.weight"]
        # print(
        #    f"Final Layer Norm - model.language_model.encoder.final_layernorm.weight - {original_size} -> {nemo_state_dict['model.language_model.encoder.final_layernorm.weight'].shape}"
        # )
        # assert nemo_state_dict["model.language_model.encoder.final_layernorm.weight"].shape == original_size
        # Output layer weight mapping
        # original_size = nemo_state_dict["model.language_model.output_layer.weight"].shape
        nemo_state_dict["model.language_model.output_layer.weight"] = model_state_dict["lm_head.weight"][
            tp_idx * EMBEDDING_DIM : (tp_idx + 1) * EMBEDDING_DIM, :
        ]
        # print(
        #    f"Output Layer - model.language_model.output_layer.weight - {original_size} -> {nemo_state_dict['model.language_model.output_layer.weight'].shape}"
        # )
        # assert nemo_state_dict["model.language_model.output_layer.weight"].shape == original_size
        # Other layer mappings
        for layer in range(TOTAL_LAYERS):
            layer_mapping = build_layer_mapping(layer)
            for k in layer_mapping.keys():
                # original_size = nemo_state_dict[k].shape
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
                elif isinstance(layer_mapping[k], torch.Tensor):
                    nemo_state_dict[k] = layer_mapping[k]
                else:
                    nemo_state_dict[k] = model_state_dict[layer_mapping[k]]
                # print(f"Layer {layer} - {k} - {original_size} -> {nemo_state_dict[k].shape}")
                # assert nemo_state_dict[k].shape == original_size
        # break view relationships otherwise pytorch will save original weights
        # to back the slices
        nemo_state_dict = {k: v.clone() for k, v in nemo_state_dict.items()}
        save_nemo_state_dict(nemo_state_dict, tp_idx)

    def get_self_attention_weight(model_state_dict, layer_mapping, key, tp_idx):
        llama_query = model_state_dict[layer_mapping[key][0]][tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM, :]
        llama_key = model_state_dict[layer_mapping[key][1]][tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM, :]
        llama_value = model_state_dict[layer_mapping[key][2]][tp_idx * PART_ATTN_DIM : (tp_idx + 1) * PART_ATTN_DIM, :]
        return torch.cat([llama_query, llama_key, llama_value], dim=0)

    def get_mlp_weight(model_state_dict, layer_mapping, key, tp_idx):
        llama_weight = model_state_dict[layer_mapping[key]]
        return llama_weight[tp_idx * PART_MLP_DIM : (tp_idx + 1) * PART_MLP_DIM, :]

    # dummy config

    megatron_cfg_path = Path(__file__).parent / "megatron_7b_llama.yaml"

    megatron_cfg = OmegaConf.load(megatron_cfg_path)
    megatron_cfg.name = f"megatron_{args.name}"
    megatron_cfg.trainer.num_nodes = 1
    megatron_cfg.trainer.devices = TOTAL_TP
    megatron_cfg.model.padded_vocab_size = config.vocab_size
    megatron_cfg.model.hidden_size = config.hidden_size
    megatron_cfg.model.ffn_hidden_size = config.intermediate_size
    megatron_cfg.model.num_layers = config.num_hidden_layers
    megatron_cfg.model.num_attention_heads = config.num_attention_heads
    megatron_cfg.model.max_position_embeddings = config.max_position_embeddings
    megatron_cfg.model.seq_length = config.max_sequence_length

    megatron_cfg.exp_manager.create_wandb_logger = False
    megatron_cfg.exp_manager.create_checkpoint_callback = False


    default_config = default_ppo_config()

    trl_config = default_config.evolve(
        train=dict(
            default_config.train.__dict__,
            checkpoint_dir=OUTPUT_FOLDER,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model=None,
                megatron_cfg=megatron_cfg,
            ),
        ),
    )

    def reward_fn(samples, **kwargs):
        return [0.0 for _ in samples]

    # Take few words off of movies reviews as prompts

    prompts = ["dummy" for _ in range(128)]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=trl_config,
    )

    return
    del nemo_model
    
    map_weights(torch.distributed.get_rank())

    print("map weights")
    #for tp in range(TOTAL_TP):
    #    map_weights(tp)

    OmegaConf.save(megatron_cfg, str(Path(OUTPUT_FOLDER) / f"megatron_{args.name}.yaml"))

    print("Done")

if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--output_folder", type=str, required=True)
    args.add_argument("--total_tp", type=int, required=True)
    args.add_argument("--name", type=str, required=True)
    main(args.parse_args())
