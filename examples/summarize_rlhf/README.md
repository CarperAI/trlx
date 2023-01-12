## Learning to summarize from Human Feedback using trlx

### Training setup:

1. Train SFT:
```bash
    cd  sft/ && deepspeed train_gptj_summarize.py
```

Checkpoint: [SFT](https://huggingface.co/pvduy/openai_summarize_sft_gptj)

2. Train Reward Model
```bash
    cd reward_model/ && deepspeed train_reward_model_gptj.py
```

Download reward model checkpoint:
```bash
    mkdir reward_model/rm_checkpoint
    wget https://huggingface.co/pvduy/openai_summarize_rm_checkpoint/resolve/main/pytorch_model.bin -O reward_model/rm_checkpoint/pytorch_model.bin
```

3. PPO training
```bash
    accelerate launch trlx_gptj_text_summarization.py
```


### Results:
On 1000 samples from CNN/DailyMail test dataset:
1. SFT vs PPO
- Rouge scores

| Model | Rouge-1 | Rouge-2 | Rouge-L | Average |
| --- | --- | --- | --- |   --- |
| SFT | 0.334 | 0.125 | 0.261 | 0.240 |
| PPO | 0.323 | 0.109 | 0.238 | 0.223 |

- Reward scores

| Model | Average Reward | Reward $\Delta$ |
| --- | --- | --- |
| SFT | 2.729 | -0.181 |
| PPO | 3.291 | 0.411 |

2. Reward model accuracy:

![image](https://user-images.githubusercontent.com/28798474/210157656-c5b20b9a-f6ef-4e88-a0ee-5596d5b28d58.png)

3. Examples of generated summaries: [here](https://wandb.ai/carperai/summarize_RLHF/runs/1rpm40g8)
