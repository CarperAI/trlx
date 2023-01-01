import torch
import evaluate
from tqdm import tqdm
import pandas as pd
from reward_model.reward_model import GPTRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def load_model(path='pvduy/openai_summarize_sft_gptj'):
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    model = AutoModelForCausalLM.from_pretrained(path)
    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return model, tokenizer



rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
rw_tokenizer.pad_token = rw_tokenizer.eos_token
rw_model = GPTRewardModel("pvduy/openai_summarize_sft_gptj")
rw_model.load_state_dict(torch.load("reward_model/rm_checkpoint/pytorch_model.bin"))
rw_model.half()
rw_model.eval()
rw_device = torch.device("cuda:{}".format(1))
rw_model.to(rw_device)


def reward_fn(samples):        
    lst_scores = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i:i+batch_size]
        sub_samples = ['<|startoftext|>' + chosen + '<|endoftext|>' for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples, 
            truncation=True, 
            max_length=550, 
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings_dict['input_ids'].to(rw_device)
        attn_masks = encodings_dict['attention_mask'].to(rw_device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        lst_scores.append(sub_scores['chosen_end_scores'])
    scores = torch.cat(lst_scores, dim=0)
    return scores

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()

    lst_pred = []
    lst_summarize = []
    lst_post = []
    rouge = evaluate.load('rouge')
    count = 0
    for post, summarize in tqdm(zip(test_post_list, test_summ_list), total=len(test_post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {'max_new_tokens': 50, 'eos_token_id': 50256, 'pad_token_id': 50256}
        summ_tokens = model.generate(
            txt_tokens,
            attention_mask=attention_mask,
            **kwargs
        )
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("TL;DR:")[1].replace("<|endoftext|>", "")
        lst_pred.append(pred)
        lst_summarize.append(summarize)
        lst_post.append(post)
        if count % 10 == 0:
            result = rouge.compute(predictions=lst_pred, references=lst_summarize)
            print(result)
        if count == 1000:
            break
        count += 1
    df = pd.DataFrame.from_dict({"pred": lst_pred, "truth": lst_summarize, "post": lst_post})
    result = rouge.compute(predictions=lst_pred, references=lst_summarize)
    print(result)
    return df

#SFT
#Reward score pred:  2.457
#Reward score truth:  2.863

def inference_batches(model, tokenizer, test_post_list, test_summ_list, batch_size=16):
    model.to("cuda")
    model.eval()

    lst_pred = []
    lst_summarize = []
    lst_post = []
    rouge = evaluate.load('rouge')

    # Iterate over the input data in mini-batches
    for i in tqdm(range(0, len(test_post_list), batch_size)):
        batch_post_list = test_post_list[i:i+batch_size]
        batch_summ_list = test_summ_list[i:i+batch_size]

        # Convert input data to tensors
        encode_dict = tokenizer(batch_post_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()

        # Perform inference on the batch
        kwargs = {'max_new_tokens': 50, 'eos_token_id': 50256, 'pad_token_id': 50256}
        summ_tokens = model.generate(
            txt_tokens,
            attention_mask=attention_mask,
            **kwargs
        )

        # Decode output tokens
        preds = tokenizer.batch_decode(summ_tokens)

        # Add predictions, truths, and input posts to lists
        lst_pred += preds
        lst_summarize += batch_summ_list
        lst_post += batch_post_list

        # Compute rouge scores every 10 mini-batches
        result = rouge.compute(predictions=lst_pred, references=lst_summarize)
        print(result)

    # Compute final rouge scores and create a dataframe
    result = rouge.compute(predictions=lst_pred, references=lst_summarize)
    print(result)
    df = pd.DataFrame.from_dict({"pred": lst_pred, "truth": lst_summarize, "post": lst_post})
    return df


if __name__=="__main__":

    model, tokenizer = load_model('pvduy/openai_summarize_ppo_gptj')


    test_post_list = [
        sample['prompt'] for sample in load_dataset("pvduy/openai_summarize_tldr", split="test")
    ]
    test_summ_list = [
        sample['label'] for sample in load_dataset("pvduy/openai_summarize_tldr", split="test")
    ]
    
    df_result = inference(model, tokenizer)
    sup_pred = df_result['pred'].values
    truth = df_result['truth'].values

    
    scores_pred = []
    scores_truth = []
    lst_preds = []
    lst_truth = []
    lst_post = []
    batch_size = 16
    for i in range(0, len(df_result), batch_size):
        predicts = df_result['pred'].values[i:i+batch_size]
        labels = df_result['truth'].values[i:i+batch_size]
        posts = df_result['post'].values[i:i+batch_size]
        data_pred = [posts[i] + predicts[i] for i in range(len(predicts))]
        data_truth = [posts[i] + labels[i] for i in range(len(labels))]
        lst_preds.extend(list(predicts))
        lst_truth.extend(list(labels))
        lst_post.extend(list(posts))
        scores_pred.extend(list(reward_fn(data_pred).cpu().numpy()))
        scores_truth.extend(list(reward_fn(data_truth).cpu().numpy()))

    df = pd.DataFrame.from_dict(
        {
            "pred": lst_preds, "truth": lst_truth, 
            "post": lst_post, "score_pred": scores_pred, "score_truth": scores_truth
        }
    )
    df.to_csv("supervised_with_reward_scores.csv", index=False)
    print("Reward score pred: ", df.score_pred.values.mean())
    print("Reward score truth: ", df.score_truth.values.mean())
