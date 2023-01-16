""" Basic tic tac toe implementation to use as a language model finetuning task. """
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from typing import List

class BoardState:
    """ A class to represent the state of a tic tac toe game """
    def __init__(self):
        self.blank = 0
        self.x = 1
        self.o = 2
        self.x_str = 'x'
        self.o_str = 'o'
        self.board_state = self.blank*np.ones((3,3))
        self.map = {self.x:'x', self.o:'o', self.blank: '-'}

    def get_valid_moves(self):
        ''' return a list of valid (i,j,player) moves '''
        # work out whose turn it is
        num_x = np.sum(self.board_state == self.x)
        num_o = np.sum(self.board_state == self.o)
        if num_x == num_o:
            turn = self.x
        elif num_x == num_o + 1:
            turn = self.o
        else:
            print("Invalid board state")

        # make list
        l = []
        for i in range(3):
            for j in range(3):
                if self.board_state[i,j] == self.blank:
                    l.append((i,j,turn))
        return l


    def make_move(self, i, j, player):
        # check if legal
        if i >= 3 or i < 0 or j >= 3 or j < 0:
            print("Index out of bounds")
        elif self.board_state[i,j] != self.blank:
            print("Not a blank square")
        elif player != self.x and player != self.o:
            print("Invalid player")

        # modify board
        self.board_state[i,j] = player


    def check_win(self):
        for player in [self.x, self.o]:
            won = False
            # check columns
            if np.any(np.all(self.board_state == player, axis=0)):
                won = True
            # check rows
            if np.any(np.all(self.board_state == player, axis=1)):
                won = True

            # check diagonals
            elif np.all(np.diag(self.board_state) == player) \
                    or np.all(np.diag(np.fliplr(self.board_state))== player):
                won = True

            if won:
                print(f"Player {player} wins!")


    def __str__(self):
        b = self.board_state
        out = ''
        # convert state to string
        for i in range(3):
            for j in range(3):
                out += f" {self.map[b[i,j]]}"
            out += "\n"
        return out


    def parse_str(self, string):
        # check if valid string
        # convert string to state
        raise NotImplementedError


def generate_random_game():
    b = BoardState()
    game_state_history = [ str(b) ]
    for t in range(9):
        valid_moves = b.get_valid_moves()
        move = np.random.choice(len(valid_moves))
        b.make_move(*valid_moves[move])
        game_state_history.append( str(b) )

    return "Let's play Tic Tac Toe:\n" + "\n".join(game_state_history)

def generate_n_games(n):
    return [ generate_random_game() for _ in range(n) ]

def generate_dataset(number_games: int) -> Dataset:
    # Create the list of games (each is a dict with one text attribute of the
    # prompt)
    games: List[str] = generate_n_games(number_games)
    games_prepared = []
    for prompt in games:
        games_prepared.append({"text": prompt})
        
    # Create dataset
    dataset = Dataset.from_dict({"train": games_prepared})
    return dataset