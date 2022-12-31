## Learning to summarize from Human Feedback using trlx


### Training setup:

1. Train SFT:
```bash
    cd  sft/ && deepspeed train_gptj_summarize.py
```

Checkpoint: [SFT](https://huggingface.co/pvduy/openai_summarize_sft_gptj)

2. Train Reward Model
```bash
    cd reward_model/ && deepspeed train_reward_gptj.py
```

Download reward checkpoint:
```bash
    mkdir reward_model/rm_checkpoint
    wget https://huggingface.co/pvduy/openai_summarize_rm_checkpoint/resolve/main/pytorch_model.bin -O reward_model/rm_checkpoint/pytorch_model.bin
```

3. PPO training
```bash
    accelerate launch trlx_gptj_text_summarization.py
```


### Results:
1. SFT vs PPO 

| Model | Rouge-1 | Rouge-2 | Rouge-L |
| --- | --- | --- | --- |
| SFT | 0.32 | 0.12 | 0.29 |
| PPO | 0.34 | 0.13 | 0.31 |

2. Reward model accuracy
![image](https://user-images.githubusercontent.com/28798474/210157656-c5b20b9a-f6ef-4e88-a0ee-5596d5b28d58.png)

3. Examples of generated summaries: [here](https://wandb.ai/pvduy/trlx/runs/1rpm40g8)
