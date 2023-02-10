## Learning to summarize from Human Feedback using `trlx`

This example shows how to use `trlx` to train a summarization model using human feedback
following the fine-tuning procedures described in Stiennon et al.'s, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)".


Before running everything, we need some extra packages not included in the `trlx` dependency list. Specifically, we need HuggingFace's [`evaluate`](https://huggingface.co/docs/evaluate/index) package and Google's re-implementation of ROUGE, [`rouge-score`](https://github.com/google-research/google-research/tree/master/rouge). To install them, run `requirements.txt` in this example's root directory:

```bash
pip install -r requirements.txt
```

### Training Process

For an in-depth description of the example, please refer to our [blog post](http://wandb.me/summarize-rlhf-trlx). We leave the following for a quick overview of the fine-tuning process and what scripts to run.


1. Train SFT:
    ```bash
    cd sft/ && deepspeed train_gptj_summarize.py
    ```
    Checkpoint: [SFT](https://huggingface.co/CarperAI/openai_summarize_tldr_sft)

2. Train Reward Model:
    ```bash
    cd reward_model/ && deepspeed train_reward_model_gptj.py
    ```
    Download reward model checkpoint:
    ```bash
    mkdir reward_model/rm_checkpoint
    wget https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin -O reward_model/rm_checkpoint/pytorch_model.bin
    ```

3. PPO training:
    ```bash
    accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py
    ```
    Checkpoint: [PPO](https://huggingface.co/CarperAI/openai_summarize_tldr_ppo)

    🩹 Warning: This particular training configuration requires at least 55GB of VRAM and is setup to use two GPUs, decrease `batch_size` in case you're running out of memory.


### Results

The following tables display ROUGE and reward scores on the test set of the TL;DR dataset between SFT and PPO models.

1. SFT vs PPO

    __ROUGE scores__

    | Model | Rouge-1 | Rouge-2 | Rouge-L | Average |
    | --- | --- | --- | --- |   --- |
    | SFT | 0.334 | 0.125 | 0.261 | 0.240 |
    | PPO | 0.323 | 0.109 | 0.238 | 0.223 |

    __Reward scores__

    | Model | Average Reward | Reward $\Delta$ |
    | --- | --- | --- |
    | SFT | 2.729 | -0.181 |
    | PPO | 3.291 | +0.411 |


2. Examples of generated summaries can be found [here](https://wandb.ai/carperai/summarize_RLHF/runs/2uirt89a).

3. Check our blog post for metric logs and other results [here](http://wandb.me/summarize-rlhf-trlx).

## References

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)", Neural Information Processing Systems, 2020.
