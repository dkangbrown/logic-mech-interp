import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
# from dotenv import load_dotenv
import gc
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache, patching
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from huggingface_hub import login

t.set_grad_enabled(False)

# Make sure exercises are in the path
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

# from plotly_utils import imshow, line, scatter, bar
# import part3_indirect_object_identification.tests as tests

# load_dotenv()

def get_prompts(full_prompt):
    prompt_word_list = full_prompt.split(' ')
    substring = ""
    prompt_list = []
    answers = []
    for word in prompt_word_list:
        if (word[:4] == 'True'):
            prompt_list.append(substring)
            answers.append(('True', 'False'))
        if (word[:5] == 'False'):
            prompt_list.append(substring)
            answers.append(('False', 'True'))
        substring += ' ' + word
    return prompt_list, answers

def get_prompts_icl(full_prompt):
    prompt_word_list = full_prompt.split('\n')
    substring = ""
    prompt_list = []
    answers = []
    for line in prompt_word_list:
        substring += line
        if (line[-5:] == 'True'):
            prompt_list.append(substring[:-4])
            answers.append(('True', 'False'))
            substring += '\n'
        if (line[-6:] == 'False'):
            prompt_list.append(substring[:-5])
            answers.append(('False', 'True'))
            substring += '\n'
        substring += '\n'
    return prompt_list, answers

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False
) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # SOLUTION
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    
    top_ids = t.argsort(final_logits, descending=True)[:, :5]

    return answer_logit_diff, top_ids if per_prompt else answer_logit_diff.mean()

def print_prompt(prompts, answers, answer_tokens):
    rprint(prompts)
    rprint(answers)
    rprint(answer_tokens)
    print(len(answers))

    table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

    for prompt, answer in zip(prompts, answers):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]))

    rprint(table)

def print_prompt_answers(prompts, tokens, answer_tokens):
    original_logits = []
    cache = []

        # Run the model and cache all activations
    print("Starting to run with cache")
    original_logits, cache = model.run_with_cache(tokens)
    print("Done running with cache")

    original_per_prompt_diff, top_ids = logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
    print("Per prompt logit difference:", original_per_prompt_diff)
    original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
    print("Average logit difference:", original_average_logit_diff)

    top_answers = [[model.tokenizer.decode(t) for t in w] for w in top_ids]
    top_answers_string = []
    for prompt_answer in top_answers:
        ans_str = ""
        for answer in prompt_answer:
            ans_str += " " + answer
        top_answers_string.append(ans_str)

    cols = [
        "Prompt",
        Column("Correct", style="rgb(0,200,0) bold"),
        Column("Incorrect", style="rgb(255,0,0) bold"),
        Column("Logit Difference", style="bold"),
        Column("Top Answers", style="bold")
    ]
    table = Table(*cols, title="Logit differences")

    for prompt, answer, logit_diff, top_answer in zip(prompts, answers, original_per_prompt_diff, top_answers_string):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}", top_answer)

    rprint(table)

    return cache

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    '''
    # SOLUTION
    batch_size = residual_stack.size(-2)
    print("performing ln to stack")
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1).cpu()
    print("performing einops")
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size

def display_logit_attr(cache, logit_diff_directions):

    accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache, logit_diff_directions)

    print("performing display")

    logit_diff_accum = px.line(
        logit_lens_logit_diffs,
        # hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        # xaxis_tickvals=labels,
        width=800
    )

    print("done calculating display")

    logit_diff_accum.write_image("images/logit_diff_accum.png")

    print("done showing")

    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache, logit_diff_directions)

    logit_diff_layer = px.line(
        per_layer_logit_diffs,
        # hovermode="x unified",
        title="Logit Difference From Each Layer",
        labels={"x": "Layer", "y": "Logit Diff"},
        # xaxis_tickvals=labels,
        width=800
    )

    logit_diff_layer.write_image("images/logit_diff_layer.png")

    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = einops.rearrange(
        per_head_residual,
        "(layer head) ... -> layer head ...",
        layer=model.cfg.n_layers
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache, logit_diff_directions)

    logit_diff_head = px.imshow(
        per_head_logit_diffs,
        labels={"x":"Head", "y":"Layer"},
        title="Logit Difference From Each Head",
        width=600
    )

    logit_diff_head.write_image("images/logit_diff_head.png")

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()


# k = 3

# for head_type in ["Positive", "Negative"]:

#     # Get the heads with largest (or smallest) contribution to the logit difference
#     top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)

#     # Get all their attention patterns
#     attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
#         cache["pattern", layer][-10:, head][0]
#          for layer, head in top_heads
#     ])

#     # Display results
#     display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
#     display(cv.attention.attention_patterns(
#         attention = attn_patterns_for_important_heads,
#         tokens = model.to_str_tokens(tokens[0]),
#         attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
#     ))

def residual_patching(model, device, tokens, prompts):

    clean_tokens = tokens
    # Swap each adjacent pair to get corrupted tokens
    # indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(tokens))]
    # corrupted_tokens = clean_tokens[indices]

    def swap_tf(prompts):
        swapped_prompts = []
        for prompt in prompts:
            swapped_prompt = ""
            lines = prompt.split('\n')
            for line in lines:
                if line == "True":
                    swapped_prompt += "False" + '\n'
                elif line == "False":
                    swapped_prompt += "True" + '\n'
                else:
                    swapped_prompt += line + '\n'
            swapped_prompts.append(swapped_prompt)
        return swapped_prompts

    corrupted_prompts = swap_tf(prompts)
    print(corrupted_prompts)
    print('calculating tokens')
    corrupted_tokens = model.to_tokens(corrupted_prompts,padding_side='left')
    print('done calculating tokens')
    clean_tokens = clean_tokens.to(device)
    corrupted_tokens = corrupted_tokens.to(device)
    print('done moving tokens')

    print(
        "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
        "Corrupted string 0:", model.to_string(corrupted_tokens[0])
    )

    print(len(clean_tokens))
    print(len(corrupted_tokens))

    gc.collect()
    t.cuda.empty_cache()

    print("Starting to run with cache")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    print("Done with clean tokens, now corrupt tokens")
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
    print("Done running with cache")

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens, per_prompt=False)[1]
    print(clean_logit_diff)
    print(clean_logit_diff.size)
    # print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)[1]
    # print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

    return clean_logit_diff, clean_tokens, corrupted_logit_diff, corrupted_tokens, clean_cache

if __name__ == "__main__":

    os.environ["HF_TOKEN"] = "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"
    login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)

    print(t.version.cuda)

    if t.cuda.is_available():
        print("cuda")
    else:
        print("cpu")

    print(t.cuda.get_device_name(0))
        
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    MAIN = __name__ == "__main__"

    model = HookedTransformer.from_pretrained(
        "gemma-2-2b",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=False,
    )

    hg06_prompt = "# Instructions\nLearn the secret rule to label the objects in groups correctly. The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. If an object in a group follows the rule, it should be labeled 'True'. Otherwise it should be labeled 'False'.\n\n# Quiz\n\n## Group 1\nlarge blue triangle\nlarge blue rectangle\nmedium yellow circle\nsmall blue circle\nlarge green circle\n\n## Group 1 Answers\nlarge blue triangle -> True\nlarge blue rectangle -> True\nmedium yellow circle -> True\nsmall blue circle -> True\nlarge green circle -> True\n\n## Group 2\nlarge yellow triangle\nmedium blue circle\n\n## Group 2 Answers\nlarge yellow triangle -> False\nmedium blue circle -> True\n\n## Group 3\nlarge yellow triangle\nsmall yellow triangle\n\n## Group 3 Answers\nlarge yellow triangle -> False\nsmall yellow triangle -> False\n\n## Group 4\nsmall blue circle\nlarge yellow rectangle\nmedium yellow rectangle\nmedium yellow triangle\n\n## Group 4 Answers\nsmall blue circle -> True\nlarge yellow rectangle -> False\nmedium yellow rectangle -> False\nmedium yellow triangle -> False\n\n## Group 5\nmedium blue circle\nmedium blue rectangle\nsmall blue triangle\n\n## Group 5 Answers\nmedium blue circle -> True\nmedium blue rectangle -> True\nsmall blue triangle -> True\n\n## Group 6\nsmall yellow triangle\n\n## Group 6 Answers\nsmall yellow triangle -> False\n\n## Group 7\nsmall yellow rectangle\nlarge blue rectangle\n\n## Group 7 Answers\nsmall yellow rectangle -> False\nlarge blue rectangle -> True\n\n## Group 8\nlarge yellow rectangle\nlarge green circle\nlarge green triangle\nsmall green rectangle\n\n## Group 8 Answers\nlarge yellow rectangle -> False\nlarge green circle -> True\nlarge green triangle -> False\nsmall green rectangle -> False\n\n## Group 9\nmedium yellow circle\nmedium yellow rectangle\nmedium blue circle\nlarge blue triangle\n\n## Group 9 Answers\nmedium yellow circle -> True\nmedium yellow rectangle -> False\nmedium blue circle -> True\nlarge blue triangle -> True\n\n## Group 10\nmedium green circle\nlarge blue circle\n\n## Group 10 Answers\nmedium green circle -> True\nlarge blue circle -> True\n\n## Group 11\nsmall blue rectangle\n\n## Group 11 Answers\nsmall blue rectangle -> True\n\n## Group 12\nsmall yellow circle\nmedium green triangle\nsmall blue triangle\nmedium green rectangle\n\n## Group 12 Answers\nsmall yellow circle -> True\nmedium green triangle -> False\nsmall blue triangle -> True\nmedium green rectangle -> False\n\n## Group 13\nmedium blue circle\nmedium yellow rectangle\nlarge yellow rectangle\nlarge blue triangle\n\n## Group 13 Answers\nmedium blue circle -> True\nmedium yellow rectangle -> False\nlarge yellow rectangle -> False\nlarge blue triangle -> True\n\n## Group 14\nmedium yellow triangle\nsmall yellow triangle\nlarge green circle\n\n## Group 14 Answers\nmedium yellow triangle -> False\nsmall yellow triangle -> False\nlarge green circle -> True\n\n## Group 15\nlarge yellow rectangle\nlarge green triangle\n\n## Group 15 Answers\nlarge yellow rectangle -> False\nlarge green triangle -> False\n\n"
    hg08_prompt = "# Instructions\nLearn the secret rule to label the objects in groups correctly. The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. If an object in a group follows the rule, it should be labeled 'True'. Otherwise it should be labeled 'False'.\n\n# Quiz\n\n## Group 1\nmedium green triangle\nsmall yellow rectangle\n\n## Group 1 Answers\nmedium green triangle -> True\nsmall yellow rectangle -> False\n\n## Group 2\nsmall green circle\nlarge blue triangle\nmedium green triangle\nsmall blue rectangle\n\n## Group 2 Answers\nsmall green circle -> True\nlarge blue triangle -> True\nmedium green triangle -> True\nsmall blue rectangle -> True\n\n## Group 3\nlarge yellow triangle\nsmall blue rectangle\n\n## Group 3 Answers\nlarge yellow triangle -> False\nsmall blue rectangle -> True\n\n## Group 4\nlarge green circle\n\n## Group 4 Answers\nlarge green circle -> True\n\n## Group 5\nlarge yellow triangle\nsmall blue rectangle\nlarge yellow circle\nlarge green circle\n\n## Group 5 Answers\nlarge yellow triangle -> False\nsmall blue rectangle -> True\nlarge yellow circle -> False\nlarge green circle -> True\n\n## Group 6\nlarge green triangle\nsmall green rectangle\nlarge green rectangle\nmedium blue rectangle\n\n## Group 6 Answers\nlarge green triangle -> True\nsmall green rectangle -> True\nlarge green rectangle -> True\nmedium blue rectangle -> True\n\n## Group 7\nmedium blue rectangle\nsmall blue circle\nmedium blue circle\nmedium yellow circle\nsmall yellow circle\n\n## Group 7 Answers\nmedium blue rectangle -> True\nsmall blue circle -> True\nmedium blue circle -> True\nmedium yellow circle -> False\nsmall yellow circle -> False\n\n## Group 8\nsmall green rectangle\nmedium yellow circle\nlarge green rectangle\nmedium green circle\n\n## Group 8 Answers\nsmall green rectangle -> True\nmedium yellow circle -> False\nlarge green rectangle -> True\nmedium green circle -> True\n\n## Group 9\nsmall green rectangle\nlarge green triangle\nsmall green triangle\n\n## Group 9 Answers\nsmall green rectangle -> True\nlarge green triangle -> True\nsmall green triangle -> True\n\n## Group 10\nsmall yellow triangle\nmedium yellow circle\nsmall blue circle\n\n## Group 10 Answers\nsmall yellow triangle -> False\nmedium yellow circle -> False\nsmall blue circle -> True\n\n## Group 11\nlarge blue triangle\nsmall yellow triangle\nlarge green circle\nmedium blue triangle\n\n## Group 11 Answers\nlarge blue triangle -> True\nsmall yellow triangle -> False\nlarge green circle -> True\nmedium blue triangle -> True\n\n## Group 12\nmedium green triangle\n\n## Group 12 Answers\nmedium green triangle -> True\n\n## Group 13\nlarge blue rectangle\nlarge green rectangle\nmedium yellow circle\n\n## Group 13 Answers\nlarge blue rectangle -> True\nlarge green rectangle -> True\nmedium yellow circle -> False\n\n## Group 14\nmedium blue rectangle\n\n## Group 14 Answers\nmedium blue rectangle -> True\n\n## Group 15\nsmall yellow triangle\nlarge green triangle\n\n## Group 15 Answers\nsmall yellow triangle -> False\nlarge green triangle -> True\n\n"
    hg09_prompt = "# Instructions\nLearn the secret rule to label the objects in groups correctly. The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. If an object in a group follows the rule, it should be labeled 'True'. Otherwise it should be labeled 'False'.\n\n# Quiz\n\n## Group 1\nlarge yellow rectangle\n\n## Group 1 Answers\nlarge yellow rectangle -> False\n\n## Group 2\nmedium green circle\nlarge blue triangle\n\n## Group 2 Answers\nmedium green circle -> False\nlarge blue triangle -> False\n\n## Group 3\nsmall blue circle\nsmall green triangle\nsmall green circle\nsmall yellow circle\nmedium blue rectangle\n\n## Group 3 Answers\nsmall blue circle -> True\nsmall green triangle -> False\nsmall green circle -> False\nsmall yellow circle -> False\nmedium blue rectangle -> False\n\n## Group 4\nlarge blue triangle\nsmall green rectangle\nmedium yellow circle\nsmall yellow rectangle\n\n## Group 4 Answers\nlarge blue triangle -> False\nsmall green rectangle -> False\nmedium yellow circle -> False\nsmall yellow rectangle -> False\n\n## Group 5\nsmall yellow rectangle\nmedium green rectangle\nmedium blue triangle\nmedium green circle\n\n## Group 5 Answers\nsmall yellow rectangle -> False\nmedium green rectangle -> False\nmedium blue triangle -> False\nmedium green circle -> False\n\n## Group 6\nmedium yellow circle\nlarge yellow rectangle\nlarge green triangle\nsmall yellow circle\n\n## Group 6 Answers\nmedium yellow circle -> False\nlarge yellow rectangle -> False\nlarge green triangle -> False\nsmall yellow circle -> False\n\n## Group 7\nlarge yellow rectangle\n\n## Group 7 Answers\nlarge yellow rectangle -> False\n\n## Group 8\nlarge blue circle\nmedium green triangle\nlarge green triangle\n\n## Group 8 Answers\nlarge blue circle -> True\nmedium green triangle -> False\nlarge green triangle -> False\n\n## Group 9\nlarge yellow circle\nmedium blue triangle\nmedium green rectangle\n\n## Group 9 Answers\nlarge yellow circle -> False\nmedium blue triangle -> False\nmedium green rectangle -> False\n\n## Group 10\nlarge blue circle\nlarge green rectangle\nlarge green circle\n\n## Group 10 Answers\nlarge blue circle -> True\nlarge green rectangle -> False\nlarge green circle -> False\n\n## Group 11\nsmall green triangle\nmedium yellow rectangle\nlarge yellow rectangle\nmedium blue circle\n\n## Group 11 Answers\nsmall green triangle -> False\nmedium yellow rectangle -> False\nlarge yellow rectangle -> False\nmedium blue circle -> True\n\n## Group 12\nlarge green triangle\nsmall yellow rectangle\nlarge blue circle\n\n## Group 12 Answers\nlarge green triangle -> False\nsmall yellow rectangle -> False\nlarge blue circle -> True\n\n## Group 13\nmedium green rectangle\nlarge blue circle\nsmall green circle\n\n## Group 13 Answers\nmedium green rectangle -> False\nlarge blue circle -> True\nsmall green circle -> False\n\n## Group 14\nsmall yellow circle\nmedium yellow triangle\nlarge yellow circle\nsmall blue rectangle\nsmall yellow triangle\n\n## Group 14 Answers\nsmall yellow circle -> False\nmedium yellow triangle -> False\nlarge yellow circle -> False\nsmall blue rectangle -> False\nsmall yellow triangle -> False\n\n## Group 15\nlarge green triangle\nsmall yellow rectangle\nsmall green rectangle\n\n## Group 15 Answers\nlarge green triangle -> False\nsmall yellow rectangle -> False\nsmall green rectangle -> False\n\n"
    hg12_prompt = "# Instructions\nLearn the secret rule to label the objects in groups correctly. The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. If an object in a group follows the rule, it should be labeled 'True'. Otherwise it should be labeled 'False'.\n\n# Quiz\n\n## Group 1\nlarge blue circle\nmedium blue rectangle\n\n## Group 1 Answers\nlarge blue circle -> False\nmedium blue rectangle -> False\n\n## Group 2\nmedium green triangle\nmedium green circle\nsmall green triangle\nsmall yellow circle\n\n## Group 2 Answers\nmedium green triangle -> True\nmedium green circle -> False\nsmall green triangle -> True\nsmall yellow circle -> False\n\n## Group 3\nsmall green rectangle\nsmall green circle\nlarge yellow triangle\n\n## Group 3 Answers\nsmall green rectangle -> True\nsmall green circle -> False\nlarge yellow triangle -> True\n\n## Group 4\nmedium blue triangle\nmedium blue rectangle\nsmall yellow circle\n\n## Group 4 Answers\nmedium blue triangle -> False\nmedium blue rectangle -> False\nsmall yellow circle -> False\n\n## Group 5\nsmall blue circle\nlarge green triangle\n\n## Group 5 Answers\nsmall blue circle -> False\nlarge green triangle -> True\n\n## Group 6\nsmall green triangle\nmedium green triangle\nmedium yellow triangle\n\n## Group 6 Answers\nsmall green triangle -> True\nmedium green triangle -> True\nmedium yellow triangle -> True\n\n## Group 7\nlarge yellow rectangle\nsmall green triangle\nlarge green rectangle\nlarge yellow circle\n\n## Group 7 Answers\nlarge yellow rectangle -> True\nsmall green triangle -> True\nlarge green rectangle -> True\nlarge yellow circle -> False\n\n## Group 8\nmedium green rectangle\nsmall green rectangle\nsmall yellow rectangle\nlarge blue rectangle\n\n## Group 8 Answers\nmedium green rectangle -> True\nsmall green rectangle -> True\nsmall yellow rectangle -> True\nlarge blue rectangle -> False\n\n## Group 9\nsmall green rectangle\nsmall yellow circle\nlarge yellow rectangle\nlarge green circle\nmedium green triangle\n\n## Group 9 Answers\nsmall green rectangle -> True\nsmall yellow circle -> False\nlarge yellow rectangle -> True\nlarge green circle -> False\nmedium green triangle -> True\n\n## Group 10\nsmall yellow triangle\nsmall blue circle\nmedium blue rectangle\nsmall green triangle\n\n## Group 10 Answers\nsmall yellow triangle -> True\nsmall blue circle -> False\nmedium blue rectangle -> False\nsmall green triangle -> True\n\n## Group 11\nsmall green circle\nsmall blue rectangle\nmedium green circle\nsmall yellow circle\n\n## Group 11 Answers\nsmall green circle -> False\nsmall blue rectangle -> False\nmedium green circle -> False\nsmall yellow circle -> False\n\n## Group 12\nmedium yellow triangle\n\n## Group 12 Answers\nmedium yellow triangle -> True\n\n## Group 13\nlarge yellow rectangle\nlarge green triangle\nmedium yellow circle\nlarge green circle\n\n## Group 13 Answers\nlarge yellow rectangle -> True\nlarge green triangle -> True\nmedium yellow circle -> False\nlarge green circle -> False\n\n## Group 14\nmedium yellow circle\nmedium green circle\nlarge green triangle\nlarge yellow rectangle\n\n## Group 14 Answers\nmedium yellow circle -> False\nmedium green circle -> False\nlarge green triangle -> True\nlarge yellow rectangle -> True\n\n## Group 15\nsmall green rectangle\nlarge blue circle\nlarge yellow triangle\nlarge yellow circle\n\n## Group 15 Answers\nsmall green rectangle -> True\nlarge blue circle -> False\nlarge yellow triangle -> True\nlarge yellow circle -> False\n\n"
    hg08_processed = "# Instructions\nLearn the secret rule to label the objects in groups correctly. The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. If an object in a group follows the rule, it should be labeled 'True'. Otherwise it should be labeled 'False'.\n\n# Quiz\n\nmedium green triangle -> True\nsmall yellow rectangle -> False\nsmall green circle -> True\nlarge blue triangle -> True\nmedium green triangle -> True\nsmall blue rectangle -> True\nlarge yellow triangle -> False\nsmall blue rectangle -> True\nlarge green circle -> True\nlarge yellow triangle -> False\nsmall blue rectangle -> True\nlarge yellow circle -> False\nlarge green circle -> True\nlarge green triangle -> True\nsmall green rectangle -> True\nlarge green rectangle -> True\nmedium blue rectangle -> True\nmedium blue rectangle -> True\nsmall blue circle -> True\nmedium blue circle -> True\nmedium yellow circle -> False\nsmall yellow circle -> False\nsmall green rectangle -> True\nmedium yellow circle -> False\nlarge green rectangle -> True\nmedium green circle -> True\nsmall green rectangle -> True\nlarge green triangle -> True\nsmall green triangle -> True\nsmall yellow triangle -> False\nmedium yellow circle -> False\nsmall blue circle -> True\nlarge blue triangle -> True\nsmall yellow triangle -> False\nlarge green circle -> True\nmedium blue triangle -> True\nmedium green triangle -> True\nlarge blue rectangle -> True\nlarge green rectangle -> True\nmedium yellow circle -> False\nmedium blue rectangle -> True\nsmall yellow triangle -> False\nlarge green triangle -> True\n"
    hg08_icl = "# Instructions\nLearn the secret rule to label the objects in groups correctly. The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. If an object in a group follows the rule, it should be labeled 'True'. Otherwise it should be labeled 'False'.\n\n# Quiz\n\nmedium green triangle\nTrue\nsmall yellow rectangle\nFalse\nsmall green circle\nTrue\nlarge blue triangle\nTrue\nmedium green triangle\nTrue\nsmall blue rectangle\nTrue\nlarge yellow triangle\nFalse\nsmall blue rectangle\nTrue\nlarge green circle\nTrue\nlarge yellow triangle\nFalse\nsmall blue rectangle\nTrue\nlarge yellow circle\nFalse\nlarge green circle\nTrue\nlarge green triangle\nTrue\nsmall green rectangle\nTrue\nlarge green rectangle\nTrue\nmedium blue rectangle\nTrue\nmedium blue rectangle\nTrue\nsmall blue circle\nTrue\nmedium blue circle\nTrue\nmedium yellow circle\nFalse\nsmall yellow circle\nFalse\nsmall green rectangle\nTrue\nmedium yellow circle\nFalse\nlarge green rectangle\nTrue\nmedium green circle\nTrue\nsmall green rectangle\nTrue\nlarge green triangle\nTrue\nsmall green triangle\nTrue\nsmall yellow triangle\nFalse\nmedium yellow circle\nFalse\nsmall blue circle\nTrue\nlarge blue triangle\nTrue\nsmall yellow triangle\nFalse\nlarge green circle\nTrue\nmedium blue triangle\nTrue\nmedium green triangle\nTrue\nlarge blue rectangle\nTrue\nlarge green rectangle\nTrue\nmedium yellow circle\nFalse\nmedium blue rectangle\nTrue\nsmall yellow triangle\nFalse\nlarge green triangle\nTrue"

    prompts, answers = get_prompts_icl(hg08_icl)
    # prompts, answers = get_prompts(hg08_prompt)
    prompts = prompts[27:30]
    answers = answers[27:30]

    # Define the answers for each prompt, in the form (correct, incorrect)
    # answers = [names[::i] for names in name_pairs for i in (1, -1)]
    # answers = [(' true', ' false'), (' true', ' false'), (' false', ' true'), (' true', ' false')]
    # Define the answer tokens (same shape as the answers)
    answer_tokens = t.concat([
        model.to_tokens(names, prepend_bos=False).T for names in answers
    ])
    tokens = model.to_tokens(prompts, padding_side='left')
    # Move the tokens to the GPU
    tokens = tokens.to(device)
    answer_tokens = answer_tokens.to(device)

    # Print two tables, one with just the prompt and the other with the model's output
    print_prompt(prompts, answers, answer_tokens)
    cache = print_prompt_answers(prompts, tokens, answer_tokens)

    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens) # [batch 2 d_model]
    print("Answer residual directions shape:", answer_residual_directions.shape)

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
    logit_diff_directions = correct_residual_directions - incorrect_residual_directions # [batch d_model]
    print(f"Logit difference directions shape:", logit_diff_directions.shape)

    # Print three images: accumulated logit diff, effect of each layer on logit diff, effect of each head on logit diff
    display_logit_attr(cache, logit_diff_directions.cpu())
    tokens = tokens.to(device)

    clean_logit_diff, clean_tokens, corrupted_logit_diff, corrupted_tokens, clean_cache = residual_patching(model, device, tokens, prompts)

    clean_logit_diff = clean_logit_diff.to(device)
    clean_tokens = clean_tokens.to(device)
    corrupted_logit_diff = corrupted_logit_diff.to(device)
    corrupted_tokens = corrupted_tokens.to(device)
    clean_cache = clean_cache.to(device)

    def patch_metric(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
        corrupted_logit_diff: float = corrupted_logit_diff,
        clean_logit_diff: float = clean_logit_diff,
    ) -> Float[Tensor, ""]:
        '''
        Linear function of logit diff, calibrated so that it equals 0 when performance is
        same as on corrupted input, and 1 when performance is same as on clean input.
        '''
        # SOLUTION
        patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)[1]
        return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)

    def display_residual_patching():
        act_patch_resid_pre = patching.get_act_patch_resid_pre(
            model = model,
            corrupted_tokens = corrupted_tokens,
            clean_cache = clean_cache,
            patching_metric = patch_metric
        )

        labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]

        px.imshow(
            act_patch_resid_pre,
            labels={"x": "Position", "y": "Layer"},
            x=labels,
            title="resid_pre Activation Patching",
            width=600
        ).write_image("images/residual_patching.png")

    def display_layer_patching():
        act_patch_block_every = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, patch_metric)

        labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]

        px.imshow(
            act_patch_block_every,
            x=labels,
            facet_col=0, # This argument tells plotly which dimension to split into separate plots
            facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
            title="Logit Difference From Patched Attn Head Output",
            labels={"x": "Sequence Position", "y": "Layer"},
            width=1000,
        ).write_image("images/layer_patching.png")
    
    display_residual_patching()
    display_layer_patching()