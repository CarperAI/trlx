## Learning to summarize from Human Feedback using `trlx`

This example shows how to use `trlx` to train a summarization model using human feedback
following the fine-tuning procedures described in Stiennon et al.'s, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2106.00987)".


Before running everything, we need some extra packages not included in the `trlx` dependency list. Specifically, we need HuggingFace's [`evaluate`](https://huggingface.co/docs/evaluate/index) package and Google's re-implementation of ROUGE, [`rouge-score`](https://github.com/google-research/google-research/tree/master/rouge). To install them, run `requirements.txt` in this example's root directory:


```bash
pip install -r requirements.txt
```


### Training setup:

1. Train SFT:
```bash
cd  sft/ && deepspeed train_gptj_summarize.py
```

Checkpoint: [SFT](https://huggingface.co/CarperAI/openai_summarize_tldr_sft)

2. Train Reward Model
```bash
cd reward_model/ && deepspeed train_reward_model_gptj.py
```

Download reward model checkpoint:
```bash
mkdir reward_model/rm_checkpoint
wget https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/blob/main/pytorch_model.bin -O reward_model/rm_checkpoint/pytorch_model.bin
```

3. PPO training
```bash
accelerate launch trlx_gptj_text_summarization.py
```

Checkpoint: [PPO](https://huggingface.co/CarperAI/openai_summarize_tldr_ppo)


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
| PPO | 3.291 | +0.411 |

2. Reward model accuracy:

![image](https://user-images.githubusercontent.com/28798474/210157656-c5b20b9a-f6ef-4e88-a0ee-5596d5b28d58.png)

3. Examples of generated summaries can be found [here](https://wandb.ai/carperai/summarize_RLHF/runs/1rpm40g8).



## References

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2106.00987)", Neural Information Processing Systems, 2020.
