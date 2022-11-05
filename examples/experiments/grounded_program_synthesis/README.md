# Interpreter Grounded Program Synthesis
*Program synthesis* is the task of automatically generating programs that solve a given task by satisfying an IO condition. In Neural Program Synthesis the synthesizer is a neural network which is a Language Model that takes in an input/output pair and tries to generate the program in the defined toy DSL's Grammar.

## Toy List Manipulation DSL Grammar
The DSL has the following grammar:
```
list_expr := list[int]
integer := -5 | -4 | -3 | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5
statement :=
          | take(list_expr,integer)
          | drop(list_expr,integer)
          | reverse(list_expr)
          | sort_asc(list_expr)
          | sort_des(list_expr)
          | add_n(list_expr,integer)
          | sub_n(list_expr,integer)
          | mul_n(list_expr,integer)
          | expand_copy(list_expr)


```
To generate training/testing data run, `python3 -m lang`. The dataset would be saved in `./dataset/train.json` and `./dataset/test.json`. To use the processed dataset refer to this [google drive link](https://drive.google.com/drive/folders/1093FlJA0MF7gh25yi4-__yU6Fj-onK1v?usp=share_link).    
Each datapoint in the dataset would look like,
```json
    {"input": "Input: [4, -2, 0, 0, 5, 5] Output: [25, 25, 20, 0, 0, -10] Function:",
    "output": "sort_des(reverse(mul_n(sort_asc(sort_asc([4, -2, 0, 0, 5, 5])),5)))"}
```
## Training with TRLX
Run `python3 -m train_trlx.py` to run the training with grounded interpreter. The `reward_fn`, would return `-1` if a sample generated is of invalid syntax. it would return `0.5` if the generated syntax is valid but doesn't satisfy IO condition. 