# flake8: noqa
import copy
import json
import random
from pathlib import Path
from pprint import pprint

from tqdm import tqdm
from transformers import AutoTokenizer


def init_random_input(len_range: int = 5, value_gen=5) -> list:
    len_gen = random.randint(2, len_range + 1)
    value_range = list(range(-value_gen, value_gen + 1))
    output = []
    for index in range(len_gen):
        value_gen = random.choice(value_range)
        output.append(value_gen)
    return output


const_integer = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


# Functions in the DSL
# Each function defines a transformation in the given DSL Grammar.
def take(input_list: list, n: int) -> list:
    return input_list[:n]


def drop(input_list: list, n: int) -> list:
    return input_list[n:]


def minimum(input_list: list) -> int:
    return min(input_list)


def maximum(input_list: list) -> int:
    return max(input_list)


def reverse(input_list: list) -> list:
    return input_list[::-1]


def sort_asc(input_list: list) -> list:
    return sorted(input_list)


def sort_des(input_list: list) -> list:
    return sorted(input_list, reverse=True)


def add_n(input_list: list, n: int) -> list:
    return [x + n for x in input_list]


def sub_n(input_list: list, n: int) -> list:
    return [x - n for x in input_list]


def mul_n(input_list: list, n: int) -> list:
    return [x * n for x in input_list]


def div_n(input_list: list, n: int) -> list:
    return [x / n for x in input_list]


def expand_copy(input_list: list) -> list:
    return input_list + input_list


# Main Production Rules for the Toy DSL.
list_manip_dsl = {
    "take": take,
    "drop": drop,
    "reverse": reverse,
    "sort_asc": sort_asc,
    "sort_des": sort_des,
    "add_n": add_n,
    "sub_n": sub_n,
    "mul_n": mul_n,
    "expand_copy": expand_copy,
}


# Use this class to execute programs written in the DSL.
class Interpreter:
    def __init__(self) -> None:
        self.parser = list_manip_dsl

    def __call__(self, statement_string: str):
        """
        Evaluation Function for the interpreter.
        args:
            statement_string (str) : Statement String
        """
        try:
            return eval(statement_string)  # Adding an exception to unparsable strings
        except:
            return "ERROR"


interpreter = Interpreter()

# TEMPLATE
# This is used to store the input, output and the function template.
# Input : List given as an input to the function.
# function_template : The atomic function in a given DSL Grammar
# Output : Transformed outut by applying function on the input.
generation_template = {"function_template": "NONE", "output": "NONE", "input": []}


# Each of the generate function is used to generate a
# template for a given function
# if chosen while sampling the dataset.
# each function takes in expressions based on the grammar and generates a template.
# Example: gen_take() generates a template for the take function.
# take function has two arguments,
# list_expression and a bounded integer(Should not be more
# than the length of the list)..


def gen_take(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(range(1, len(expr1) - 1))

    formatted_fn = f"take({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_drop(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(range(1, len(expr1) - 1))

    formatted_fn = f"drop({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_minimum(expr1=None):
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"minimum({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_maximum(expr1=None):
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"maximum({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_reverse(expr1=None):
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"reverse({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_sort_asc(expr1=None):
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"sort_asc({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_sort_des(expr1=None):
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"sort_des({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_add_n(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"add_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_sub_n(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"sub_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_mul_n(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"mul_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_div_n(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"div_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_expand_copy(expr1=None, expr2=None):
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(range(1, 3))

    formatted_fn = f"expand_copy({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


list_manip_dsl_gen = {
    "take": gen_take,
    "drop": gen_drop,
    "minimum": gen_minimum,
    "maximum": gen_maximum,
    "reverse": gen_reverse,
    "sort_asc": gen_sort_asc,
    "sort_des": gen_sort_des,
    "add_n": gen_add_n,
    "sub_n": gen_sub_n,
    "mul_n": gen_mul_n,
    "div_n": gen_div_n,
    "expand_copy": gen_expand_copy,
}


class Sampler:
    def __init__(
        self,
        max_sample_length: int = 5,
        code_sep: str = ";",
        interpreter_sep: str = "->",
    ):
        self.max_sample_length = max_sample_length
        self.parser = Interpreter()
        self.production_list = list_manip_dsl
        self.production_idt = [i for i in self.production_list.keys()]
        self.production_gen_list = list_manip_dsl_gen
        self.code_sep = code_sep
        self.interpreter_sep = interpreter_sep

    def sample_production(self, gen_length: int = 5):
        init_flag = True
        hash_functions = []
        if gen_length == None:
            gen_length = self.max_sample_length

        for ind in range(gen_length):
            if init_flag:
                random_chosen_function = random.choice(self.production_idt)
                generated_function = self.production_gen_list[random_chosen_function]()
                hash_functions.append(generated_function)
                init_flag = False
            else:
                random_chosen_function = random.choice(self.production_idt)
                generated_function = self.production_gen_list[random_chosen_function](
                    hash_functions[-1]["function_template"]
                )
                if generated_function["output"] == "ERROR":
                    break
                hash_functions.append(generated_function)

        return hash_functions


def create_synthetic_dataset(size: int, io_size=3) -> dict:
    output_list = []
    sampler = Sampler()
    for i in tqdm(range(size)):
        try:
            sampled = sampler.sample_production()
            inp = sampled[0]["input"][0]
            out = sampled[-1]["output"]
            function = sampled[-1]["function_template"]
            prompt_inp = f"Input: {inp} Output: {out} Function:"
            prompt_out = function
            if out != [] and out != "ERROR":
                output_list.append(
                    {
                        "input": prompt_inp,
                        "output": prompt_out,
                        "io_inp": inp,
                        "io_out": out,
                    }
                )
        except:
            pass

    return output_list


def write_to_json(data: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)


def basic_stats(dataset, tokenizer):
    """
    Basic stats to calculate the token length of the dataset.
    """
    length_list = []
    for examples in tqdm(dataset):
        datapoint = tokenizer(examples["input"] + " " + examples["output"] + "<|endoftext|>")
        length_list.append(len(datapoint["input_ids"]))
    return {
        "max": max(length_list),
        "min": min(length_list),
        "mean": sum(length_list) / len(length_list),
    }


if __name__ == "__main__":
    # sampler = Sampler()
    # pprint(sampler.sample_production())
    # pprint(interpreter("div_n(reverse([-2, -5, -4]),1)"))
    train_data = create_synthetic_dataset(2000000)
    test_data = create_synthetic_dataset(2_000)
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    Path("dataset").mkdir(parents=True, exist_ok=True)
    write_to_json(train_data, "dataset/train.json")
    write_to_json(test_data, "dataset/test.json")
