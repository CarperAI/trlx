from collections.abc import Iterable

import torch
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.megatron.utils import (
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    forward_step,
    get_batch,
    receive_generate_info,
    repetition_penalty,
    send_generate_info,
    switch,
    tokenize_batch,
    top_k_logits,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
)
from nemo.utils import AppState


def synced_generate(
    model,
    context_tokens_tensor,
    context_length_tensor,
    task_ids,
    tokens_to_generate,
    all_probs,
    temperature,
    top_k=0,
    top_p=0.0,
    greedy=False,
    repetition_penalty=1.2,
    min_tokens_to_generate=0,
):
    context_length = context_length_tensor.min().item()
    tokenizer = model.tokenizer
    tokens, attention_mask, position_ids = get_batch(
        model, tokenizer, context_tokens_tensor
    )
    batch_token_iterator = sample_sequence_batch(
        model,
        context_tokens_tensor,
        context_length_tensor,
        task_ids,
        attention_mask,
        position_ids,
        tokens_to_generate,
        all_probs,
        temperature=temperature,
        extra={
            "top_p": top_p,
            "top_k": top_k,
            "greedy": greedy,
            "repetition_penalty": repetition_penalty,
            "min_tokens_to_generate": min_tokens_to_generate,
        },
    )

    for tokens, lengths, output_logits, full_logits in batch_token_iterator:
        context_length += 1

    if parallel_state.is_pipeline_last_stage():
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        torch.distributed.broadcast(output_logits, src, group)
        if all_probs:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            torch.distributed.broadcast(full_logits, src, group)

    else:
        if parallel_state.is_pipeline_first_stage():
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            output_logits = torch.empty(
                tokens.size(0),
                context_length - 1,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            torch.distributed.broadcast(output_logits, src, group)

            if all_probs:
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                full_logits = torch.empty(
                    tokens.size(0),
                    context_length - 1,
                    model.padded_vocab_size,
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                )
                torch.distributed.broadcast(full_logits, src, group)
    if tokens is not None:
        return tokens[:, :context_length], output_logits, full_logits


def generate(
    model,
    inputs=None,
    task_ids=None,
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
    repetition_penalty=1.0,
    min_tokens_to_generate=0,
) -> OutputType:
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        task_ids (Tensor): used to specify that task when generating with p-tuned/prompt-tuned models (optional, default=None)
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        add_BOS (bool): add the bos token at the begining of the prompt
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:
            sentences: List[str], output sentences
            tokens: List[List[str]], output sentences borken into tokens
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
            offsets: List[List[int]]  # list of tokens start positions in text
    """
    model.eval()
    tokenizer = model.tokenizer
    if torch.distributed.get_rank() == 0:
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = tokenize_batch(
                tokenizer, inputs, tokens_to_generate, add_BOS
            )
        if task_ids is None:
            # Make a dummy tensor of -1s that won't be used during generation
            task_ids = torch.neg(
                torch.ones(context_tokens_tensor.size(0), dtype=torch.int64)
            )
            task_ids = task_ids.to(device=context_tokens_tensor.get_device())

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            task_ids,
            tokens_to_generate,
            all_probs,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
        )
    else:
        (
            context_length_tensor,
            context_tokens_tensor,
            task_ids,
            tokens_to_generate,
            all_probs,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
        ) = receive_generate_info()

    output = synced_generate(
        model,
        context_tokens_tensor,
        context_length_tensor,
        task_ids,
        tokens_to_generate,
        all_probs,
        temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        min_tokens_to_generate=min_tokens_to_generate,
    )
    if output is not None:
        decode_tokens, output_logits, full_logits = output
        resp_sentences = []
        resp_sentences_seg = []

        decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)
            if not isinstance(tokenizer, TabularTokenizer):
                words = []
                for token in decode_token:
                    # Skip any soft prompt pseudo tokens
                    if token not in tokenizer.tokenizer.decoder:
                        continue
                    word = tokenizer.tokenizer.decoder[token]
                    word = bytearray(
                        [tokenizer.tokenizer.byte_decoder[c] for c in word]
                    ).decode("utf-8", errors="replace")
                    words.append(word)
                resp_sentences_seg.append(words)
            else:
                words = tokenizer.text_to_tokens(sentence)
                resp_sentences_seg.append(words)
        # offsets calculation
        all_offsets = []
        for item in resp_sentences_seg:
            offsets = [0]
            for index, token in enumerate(item):
                if index != len(item) - 1:
                    offsets.append(len(token) + offsets[-1])
            all_offsets.append(offsets)

        output = {}
        output["sentences"] = resp_sentences
        output["tokens"] = resp_sentences_seg
        output["logprob"] = output_logits
        output["full_logprob"] = full_logits
        output["token_ids"] = decode_tokens
        output["offsets"] = all_offsets
        return output


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    context_tokens,
    context_lengths,
    task_ids,
    attention_mask,
    position_ids,
    tokens_to_generate,
    all_probs=False,
    type_ids=None,
    temperature=None,
    extra={},
):
    # Importing here to avoid circular import errors
    from nemo.collections.nlp.models.language_modeling import (
        MegatronGPTPromptLearningModel,
    )

    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    tokenizer = model.tokenizer
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        if isinstance(model, MegatronGPTPromptLearningModel):
            if maxlen > model.frozen_model.cfg.encoder_seq_length + 1:
                maxlen = model.frozen_model.cfg.encoder_seq_length + 1
        else:
            if maxlen > model.cfg.encoder_seq_length + 1:
                maxlen = model.cfg.encoder_seq_length + 1

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length < maxlen:
            # types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

            attention_mask_repeat = torch.concat(
                [attention_mask for _ in range(micro_batch_size)]
            )
            setkey_value_array = torch.tensor(
                [set_inference_key_value_memory] * micro_batch_size,
                device=torch.cuda.current_device(),
            )
            len_array = torch.tensor(
                [maxlen] * micro_batch_size, device=torch.cuda.current_device()
            )

            # Only prompt learning models will have a prompt table, and require task ids
            if isinstance(model, MegatronGPTPromptLearningModel):
                batch = [
                    tokens2use,
                    attention_mask_repeat,
                    positions2use,
                    task_ids,
                    setkey_value_array,
                    len_array,
                ]
                tensor_shape = [
                    tokens2use.shape[1],
                    micro_batch_size,
                    model.frozen_model.cfg.hidden_size,
                ]
            else:
                batch = [
                    tokens2use,
                    attention_mask_repeat,
                    positions2use,
                    setkey_value_array,
                    len_array,
                ]
                tensor_shape = [
                    tokens2use.shape[1],
                    micro_batch_size,
                    model.cfg.hidden_size,
                ]

            print(f"{context_length=} {maxlen=}")

            if parallel_state.is_pipeline_first_stage():
                print("first before fwd")

            output = forward_step(model, batch, tensor_shape)

            if parallel_state.is_pipeline_first_stage():
                print("first after fwd")

            if parallel_state.is_pipeline_last_stage():
                output = output[0]["logits"].float()
                output = tensor_parallel.gather_from_tensor_model_parallel_region(
                    output
                )
                assert output is not None
                output = output.float()
                logits = output[:, -1].view(batch_size, -1).contiguous()
                print(f"{logits.shape=} {context_length=}")
                # make sure it will generate at least min_length
                min_length = extra.get("min_tokens_to_generate", 0)
                if min_length > 0:
                    within_min_length = (context_length - context_lengths) < min_length
                    logits[within_min_length, eod_id] = -float("Inf")

                # make sure it won't sample outside the vocab_size range
                logits[:, tokenizer.vocab_size :] = -float("Inf")

                if extra.get("greedy", False):
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    # handle repetition penality
                    logits = repetition_penalty(
                        logits,
                        extra.get("repetition_penalty", 1.2),
                        all_generated_indices,
                    )
                    logits = top_k_logits(
                        logits,
                        top_k=extra.get("top_k", 0),
                        top_p=extra.get("top_p", 0.9),
                    )
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)
                started = context_lengths <= context_length

                # Clamp the predicted out of vocabulary tokens
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # Replace sampled tokens w/ done token if EOD has already been sampled
                new_tokens = switch(new_tokens, eod_id, is_done)

                # Replace special soft prompt token ids with unk token ids
                if isinstance(model, MegatronGPTPromptLearningModel):
                    pseudo_token_ids_start = model.pseudo_token_ids_start
                    new_tokens[
                        (new_tokens >= pseudo_token_ids_start)
                    ] = tokenizer.unk_id
                    tokens[:, :context_length][
                        (tokens[:, :context_length] >= pseudo_token_ids_start)
                    ] = tokenizer.unk_id

                # Insert either new predicted or next prompt token
                tokens[:, context_length] = new_tokens

                if output_logits is None:
                    output = F.log_softmax(output[:, :context_length, :], 2)
                    indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                    output_logits = torch.gather(output, 2, indices).squeeze(2)
                    all_generated_indices = indices[:, :, 0]
                    if all_probs:
                        full_logits = output
                else:
                    output = F.log_softmax(output, 2)
                    indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                    new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                    # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                    output_logits = torch.cat([output_logits, new_output_logits], 1)
                    all_generated_indices = torch.cat(
                        [all_generated_indices, indices[:, :, 0]], 1
                    )
                    if all_probs:
                        full_logits = torch.cat([full_logits, output], 1)

                print(f"{new_tokens=}")
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)
                print("distributed broadcast embs last stage")
                done_token = (prev == eod_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                print("distributed broadcast done last stage")

                if all_probs:
                    yield tokens, lengths, output_logits, full_logits
                else:
                    yield tokens, lengths, output_logits, None

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    print("waiting for broadcast embs in first stage recv")
                    import traceback

                    traceback.print_stack()
                    torch.distributed.broadcast(new_tokens, src, group)
                    print("distributed broadcast embs in first stage recv")
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                print("distributed broadcast done in non first stage recv")

            print(f"{context_length=} {done=}")
            context_length += 1
            counter += 1
            if done:
                break
