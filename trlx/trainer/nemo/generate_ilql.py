from collections.abc import Iterable

import torch
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.megatron.utils import (
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import (
    model_inference_strategy_dispatcher,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    receive_generate_info,
    repetition_penalty,
    send_generate_info,
    switch,
    top_k_logits,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
)
from nemo.utils import AppState


def generate(
    model,
    inputs=None,
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
    repetition_penalty=1.0,
    min_tokens_to_generate=0,
    **strategy_args,
) -> OutputType:
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        add_BOS (bool): add the bos token at the begining of the prompt
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
        strategy_args, the extra arguments are treated as inference strategy arguments
    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:
            sentences: List[str], output sentences
            tokens: List[List[str]], output sentences borken into tokens
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
            offsets: List[List[int]]  # list of tokens start positions in text
    """
    if "strategy" in strategy_args:
        inference_strategy = strategy_args["strategy"]
    else:
        inference_strategy = model_inference_strategy_dispatcher(model, **strategy_args)
    tokenizer = model.tokenizer
    if torch.distributed.get_rank() == 0:
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        else:
            (
                context_tokens_tensor,
                context_length_tensor,
            ) = inference_strategy.tokenize_batch(inputs, tokens_to_generate, add_BOS)

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
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
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        min_tokens_to_generate=min_tokens_to_generate,
    )
    special_tokens = set()
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None:
        special_tokens.add(tokenizer.pad_token)
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        special_tokens.add(tokenizer.eos_token)
    if hasattr(tokenizer, "bos_token") and tokenizer.bos_token is not None:
        special_tokens.add(tokenizer.bos_token)
    if hasattr(tokenizer, "cls_token") and tokenizer.cls_token is not None:
        special_tokens.add(tokenizer.cls_token)
    if hasattr(tokenizer, "unk_token") and tokenizer.unk_token is not None:
        special_tokens.add(tokenizer.unk_token)
    if hasattr(tokenizer, "sep_token") and tokenizer.sep_token is not None:
        special_tokens.add(tokenizer.sep_token)
    if hasattr(tokenizer, "mask_token") and tokenizer.mask_token is not None:
        special_tokens.add(tokenizer.mask_token)
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
                    if not isinstance(token, Iterable):
                        token = [token]
                    word = tokenizer.ids_to_tokens(token)
                    if isinstance(word, Iterable):
                        word = word[0]
                    if hasattr(tokenizer.tokenizer, "byte_decoder"):
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
                    if token in special_tokens:
                        offsets.append(offsets[-1])
                    else:
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


def synced_generate(
    model,
    inference_strategy,
    context_tokens_tensor,
    context_length_tensor,
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
    batch_token_iterator = sample_sequence_batch(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
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


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    tokens_to_generate,
    all_probs=False,
    type_ids=None,
    temperature=None,
    extra={},
):
    # Importing here to avoid circular import errors

    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert (
        model.cfg.get("sequence_parallel", False) == False
    ), "sequence_parallel should be False during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint"
    assert (
        model.cfg.get("activations_checkpoint_granularity", None) is None
    ), "activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint"
    assert (
        model.cfg.get("activations_checkpoint_method", None) is None
    ), "activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint"

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_length = context_lengths.min().item()
        inference_strategy.init_batch(context_tokens, context_length)
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

        maxlen = inference_strategy.clip_max_len(maxlen)

        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        while context_length < maxlen:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens, maxlen, micro_batch_size, counter, context_length
            )
            output = inference_strategy.forward_step(batch, tensor_shape)

            if parallel_state.is_pipeline_last_stage():
                output = output[0]["logits"].float()
                output = tensor_parallel.gather_from_tensor_model_parallel_region(
                    output
                )
                assert output is not None
                output = output.float()
                logits = output[:, -1].view(batch_size, -1).contiguous()

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

                # post process the inference tokens based on the strategy
                inference_strategy.post_process(tokens, new_tokens, context_length)

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

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eod_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if all_probs:
                    yield tokens, lengths, output_logits, full_logits
                else:
                    yield tokens, lengths, output_logits, None

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break
