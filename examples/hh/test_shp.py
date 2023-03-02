import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large").cuda()
model.eval()


def shp_reward(samples, prompts, outputs):
    rm_inputs = [
        f"POST: {prompt} \n\n RESPONSE A: {output}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE"
        for prompt, output in zip(prompts, outputs)
    ]

    rm_inputs = tokenizer(rm_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**rm_inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    sscores = torch.softmax(outputs.scores[0], dim=-1)[:, 71]
    return scores


print(
    shp_reward(
        None,
        [
            "hello",
            "Instacart gave me 50 pounds of limes instead of 5 pounds... what the hell do I do with 50 pounds of limes? I've already donated a bunch and gave a bunch away. I'm planning on making a bunch of lime-themed cocktails, but... jeez. Ceviche?",
        ],
        ["world", "Lime juice, and zest, then freeze in small quantities."],
    )
)
