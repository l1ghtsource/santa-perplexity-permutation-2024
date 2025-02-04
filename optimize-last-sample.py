# refer to @woosungyoon

import gc
import os
import time
import math
import copy
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch
import heapq
from tqdm import tqdm

import matplotlib.pyplot as plt
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pprint import pprint

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HuggingFaceModelLoader:
    def __init__(self, model_path: str, load_in_8bit: bool, device_map: str):
        self.model_path = model_path
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map

    def load_model(self) -> transformers.PreTrainedModel:
        if self.load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires a CUDA device')

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map=self.device_map,
                attn_implementation="sdpa"
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
                device_map=self.device_map
            )

        model.eval()
        return model


class HuggingFaceTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
        self.bos_token = self.tokenizer.bos_token or self.tokenizer.cls_token
        self.eos_token = self.tokenizer.eos_token or self.tokenizer.sep_token
        if self.bos_token is None:
            self.bos_token = ""
        if self.eos_token is None:
            self.eos_token = ""

    def tokenize(self, texts: List[str]) -> dict:
        processed_texts = []

        for text in texts:
            combined_text = f"{self.bos_token}{text}{self.eos_token}"
            processed_texts.append(combined_text)

        model_inputs = self.tokenizer(
            processed_texts,
            return_tensors='pt',
            add_special_tokens=False,
            padding=True
        )

        if 'token_type_ids' in model_inputs:
            model_inputs.pop('token_type_ids')

        return model_inputs


class PerplexityCalculator:
    def __init__(self, model_loader, tokenizer, exp_mode=False):
        self.model = model_loader.load_model()
        self.model.eval()
        self.tokenizer = tokenizer
        self.exp_mode = exp_mode
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def get_perplexity(
        self,
        input_texts: Union[str, List[str]]
    ) -> Union[float, List[float]]:

        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        with torch.no_grad():
            input_batch = input_texts
            model_inputs = self.tokenizer.tokenize(input_batch)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

            output = self.model(**model_inputs, use_cache=False)
            logits = output['logits']

            label = model_inputs['input_ids']
            if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is not None:
                label[label == self.model.config.pad_token_id] = PAD_TOKEN_LABEL_ID

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label[..., 1:].contiguous()

            token_loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(len(logits), -1)

            valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
            sequence_loss = torch.sum(token_loss, -1) / valid_length
            loss_list.extend(sequence_loss.cpu().tolist())

        if self.exp_mode:
            ppl = [exp(i) for i in loss_list]
        else:
            ppl = loss_list

        return ppl[0] if single_input else ppl
        
        
model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
model_loader = HuggingFaceModelLoader(model_path=model_path, load_in_8bit=False, device_map='auto')
tokenizer = HuggingFaceTokenizer(model_path)
scorer = PerplexityCalculator(model_loader, tokenizer, exp_mode=False)

class Configuration:
    """
    A class holding:
      - stopwords (set)
      - semi_free_words (set)
      - config dict:
        {
          'letters': {
            letter_name: {
               'rooms': [ [occupant1, occupant2, ...], ... ]
            },
            ...
          },
          'free_pool': [some stopwords or other items]
        }
    """
    def __init__(self, stopwords=None, semi_free_words=None, initial_config=None):
        self.stopwords = set(stopwords) if stopwords else set()
        self.semi_free_words = set(semi_free_words) if semi_free_words else set()

        if initial_config:
            self.config = initial_config
        else:
            self.config = {
                "letters": {},
                "free_pool": []
            }

    def flatten_encode(self) -> str:
        tokens = []
        tokens.extend(self.config['free_pool'])
        letters_ordered = sorted(self.config['letters'])
        max_room_count = 0
        for lt in letters_ordered:
            max_room_count = max(max_room_count, len(self.config['letters'][lt]['rooms']))

        for room_idx in range(max_room_count):
            for letter in letters_ordered:
                rooms = self.config['letters'][letter]['rooms']
                if room_idx < len(rooms):
                    tokens.extend(rooms[room_idx])

        return ' '.join(tokens)

    def __repr__(self) -> str:
        return self.flatten_encode()
        

sample_5_state = {
    'letters': {
        'a': {
            'rooms': [
                ['advent', 'angel'],
                []
                ]
            },
        'b': {
            'rooms': [
                ['bake', 'beard', 'believe', 'bow'],
                []
                ]
            },
        'c': {
            'rooms': [
                ['candle', 'candy', 'card', 'carol', 'cheer', 'cheer', 'chimney', 'chimney', 'chocolate', 'cookie'],
                []
                ]
            },
        'd': {
            'rooms': [
                ['decorations', 'doll', 'dream', 'drive'],
                []
                ]
            },
        'e': {
            'rooms': [
                ['eat', 'eggnog', 'elf'],
                []
                ]
            },
        'f': {
            'rooms': [
                ['family', 'fireplace', 'fireplace', 'fruitcake'],
                []
                ]
            },
        'g': {
            'rooms': [
                ['game', 'gifts', 'gingerbread', 'give', 'greeting', 'grinch'],
                []
                ]
            },
        'h': {
            'rooms': [
                ['hohoho', 'holiday', 'holly', 'hope'],
                []
                ]
            },
        'j': {
            'rooms': [
                [ 'jingle', 'joy', 'jump'],
                []
                ]
            },
        'k': {
            'rooms': [
                ['kaggle'],
                []
                ]
            },
        'l': {
            'rooms': [
                ['laugh'],
                []
                ]
            },
        'm': {
            'rooms': [
                ['magi', 'merry', 'milk', 'mistletoe'],
                []
                ]
            },
        'n': {
            'rooms': [
                ['naughty', 'nice', 'night', 'night', 'nutcracker'],
                []
                ]
            },
        'o': {
            'rooms': [
                ['ornament', 'ornament'],
                []
                ]
            },
        'p': {
            'rooms': [
                ['paper', 'peace', 'peppermint', 'poinsettia', 'polar', 'puzzle'],
                []
                ]
            },
        'r': {
            'rooms': [
                ['reindeer', 'relax'],
                []
                ]
            },
        's': {
            'rooms': [
                ['scrooge', 'season', 'sing', 'sleep', 'sleigh', 'snowglobe', 'star', 'stocking'],
                []
                ]
            },
        't': {
            'rooms': [
                ['toy'],
                []
                ]
            },
        'u': {
            'rooms': [
                ['unwrap'],
                []
                ]
            },
        'v': {
            'rooms': [
                ['visit'],
                []
                ]
            },
        'w': {
            'rooms': [
                ['walk', 'wish', 'wonder', 'workshop', 'workshop', 'wrapping', 'wreath'],
                []
                ]
            },
        'y': {
            'rooms': [
                ['yuletide'],
                []
                ]
            },
        },
    'free_pool': ['the', 'the', 'the', 'of', 'of', 'and', 'to', 'in', 'and', 'is', 'and', 'you', 'that', 'it', 'we', 'with', 'from', 'have', 'not', 'as' ]
    }

stopwords = ['the', 'the', 'the', 'of', 'of', 'and', 'to', 'in', 'and', 'is', 'and', 'you', 'that', 'it', 'we', 'with', 'from', 'have', 'not', 'as' ]

sample5_config = Configuration(stopwords=stopwords, initial_config=sample_5_state)


class MoveRegistry:
    """
    Holds a collection of moves, each with a weight (probability).
    Each move is a function: move_func(config: Configuration) -> None
    """
    def __init__(self):
        self.moves: List[Tuple[str, Callable[[Configuration], None], float]] = []

    def register_move(self, name: str, func: Callable[[Configuration], None], weight: float):
        self.moves.append((name, func, weight))

    def unregister_move(self, name: str):
        self.moves = [(n, f, w) for (n, f, w) in self.moves if n != name]

    def pick_move(self) -> Callable[[Configuration], None]:
        """Select one move function according to the stored weights."""
        if not self.moves:
            return lambda c: None
        funcs = [m[1] for m in self.moves]
        weights = [m[2] for m in self.moves]
        chosen_func = random.choices(funcs, weights=weights, k=1)[0]
        return chosen_func
        

class Moves:
    @staticmethod
    def move_item_between_groups(cfg: Configuration):
        letters = list(cfg.config['letters'])
        if not letters:
            return
        letter_1, letter_2 = random.sample(letters, k=2)
        
        rooms_1 = cfg.config['letters'][letter_1]['rooms']
        rooms_2 = cfg.config['letters'][letter_2]['rooms']
        non_empty_1 = [idx for idx, r in enumerate(rooms_1) if r]
        if not non_empty_1:
            return
            
        src = random.choice(non_empty_1)
        dst = random.choice(list(range(len(rooms_2))))
        
        item_idx = random.randrange(len(rooms_1[src]))
        item = rooms_1[src].pop(item_idx)
        insert_pos = random.randint(0, len(rooms_2[dst]))
        rooms_2[dst].insert(insert_pos, item)
        
    @staticmethod
    def rotate_room_elements(cfg: Configuration):
        letters = list(cfg.config['letters'])
        if not letters:
            return
        letter = random.choice(letters)
        rooms = cfg.config['letters'][letter]['rooms']
        if not rooms:
            return
        r_idx = random.randrange(len(rooms))
        rlist = rooms[r_idx]
        if len(rlist) > 1:
            first_item = rlist.pop(0)
            rlist.append(first_item)

    @staticmethod
    def swap_elements_in_room(cfg: Configuration):
        letters = list(cfg.config['letters'])
        if not letters:
            return
        letter = random.choice(letters)
        rooms = cfg.config['letters'][letter]['rooms']
        swappable = [idx for idx, rlist in enumerate(rooms) if len(rlist) >= 2]
        if not swappable:
            return
        chosen_room_idx = random.choice(swappable)
        rlist = rooms[chosen_room_idx]
        i, j = random.sample(range(len(rlist)), 2)
        rlist[i], rlist[j] = rlist[j], rlist[i]

    @staticmethod
    def move_item_between_rooms(cfg: Configuration):
        letters = list(cfg.config['letters'])
        if not letters:
            return
        letter = random.choice(letters)
        rooms = cfg.config['letters'][letter]['rooms']
        if len(rooms) < 2:
            return
        non_empty = [idx for idx, r in enumerate(rooms) if r]
        if not non_empty:
            return
        src = random.choice(non_empty)
        targets = [i for i in range(len(rooms)) if i != src]
        dst = random.choice(targets)
        item_idx = random.randrange(len(rooms[src]))
        item = rooms[src].pop(item_idx)
        insert_pos = random.randint(0, len(rooms[dst]))
        rooms[dst].insert(insert_pos, item)

    @staticmethod
    def move_item_to_matching_group(cfg: Configuration):
        letters = list(cfg.config['letters'])
        if not letters:
            return
        letter_1 = random.choice(letters)
        rooms_1 = cfg.config['letters'][letter_1]['rooms']
        non_empty_1 = [idx for idx, r in enumerate(rooms_1) if r]
        if not non_empty_1:
            return
        src = random.choice(non_empty_1)
        item_idx = random.randrange(len(rooms_1[src]))
        item = rooms_1[src].pop(item_idx)
        first_letter = item[0].lower()
        if first_letter not in cfg.config['letters']:
            return
        rooms_2 = cfg.config['letters'][first_letter]['rooms']
        dst = random.choice(list(range(len(rooms_2))))
        insert_pos = random.randint(0, len(rooms_2[dst]))
        rooms_2[dst].insert(insert_pos, item)
        

    @staticmethod
    def swap_items_in_free_pool(cfg: Configuration):
        """
        Swap two items at random indices in the free_pool, if the free_pool has >= 2 items.
        """
        pool = cfg.config['free_pool']
        if len(pool) < 2:
            return
        i, j = random.sample(range(len(pool)), 2)
        pool[i], pool[j] = pool[j], pool[i]


    @staticmethod
    def free_to_room(cfg: Configuration):
        if not cfg.config['free_pool']:
            return
        item_idx = random.randrange(len(cfg.config['free_pool']))
        item = cfg.config['free_pool'].pop(item_idx)

        letters = list(cfg.config['letters'])
        if not letters:
            cfg.config['free_pool'].append(item)
            return
        letter = random.choice(letters)
        rooms = cfg.config['letters'][letter]['rooms']
        if not rooms:
            cfg.config['free_pool'].append(item)
            return
        r_idx = random.randrange(len(rooms))
        insert_pos = random.randint(0, len(rooms[r_idx]))
        rooms[r_idx].insert(insert_pos, item)

    @staticmethod
    def room_to_free(cfg: Configuration):
        stopwords = cfg.stopwords
        letters = list(cfg.config['letters'])
        if not letters:
            return
        possible_spots = []
        for letter in letters:
            rooms = cfg.config['letters'][letter]['rooms']
            for i, rlist in enumerate(rooms):
                if any((x in stopwords) for x in rlist):
                    possible_spots.append((letter, i))
        if not possible_spots:
            return
        letter, r_idx = random.choice(possible_spots)
        rlist = cfg.config['letters'][letter]['rooms'][r_idx]
        candidates = [idx for idx, x in enumerate(rlist) if (x in stopwords)]
        if not candidates:
            return
        sw_idx = random.choice(candidates)
        sw = rlist.pop(sw_idx)
        insert_pos = random.randint(0, len(cfg.config['free_pool']))
        cfg.config['free_pool'].insert(insert_pos, sw)

    @staticmethod
    def move_item_to_matching_room(cfg: Configuration):
        letters = list(cfg.config['letters'])
        if not letters:
            return
        current_letter = random.choice(letters)
        rooms = cfg.config['letters'][current_letter]['rooms']
        non_empty_rooms = [idx for idx, room in enumerate(rooms) if room]
        if not non_empty_rooms:
            return
        room_idx = random.choice(non_empty_rooms)
        room = rooms[room_idx]
        if not room:
            return
        item_idx = random.randrange(len(room))
        item = room.pop(item_idx)
        correct_letter = item[0].lower()
        
        if correct_letter == current_letter:
            room.insert(item_idx, item)
            return
        
        if correct_letter not in cfg.config['letters']:
            cfg.config['free_pool'].append(item)
            return
        
        correct_rooms = cfg.config['letters'][correct_letter]['rooms']
        if not correct_rooms:
            cfg.config['free_pool'].append(item)
            return
        
        target_room_idx = random.randrange(len(correct_rooms))
        target_room = correct_rooms[target_room_idx]
        insert_pos = random.randint(0, len(target_room))
        target_room.insert(insert_pos, item)
        
        
register = MoveRegistry()
register.register_move('rotate', Moves.rotate_room_elements, 0.2)
register.register_move('swap', Moves.swap_elements_in_room, 0.5)
register.register_move('move', Moves.move_item_between_rooms, 1.0)
register.register_move('group', Moves.move_item_between_groups, 0.5)
register.register_move('back', Moves.move_item_to_matching_room, 0.2)
# register.register_move('swap2',Moves.swap_items_in_free_pool, 0.2)
# register.register_move('f2r',Moves.free_to_room, 0.2)
# register.register_move('r2f',Moves.room_to_free, 0.2)


class SimulatedAnnealing:
    def __init__(self, start_temp, end_temp, max_iterations, cost_fn):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.max_iterations = max_iterations
        self.cost_fn = cost_fn
        self.register = register

    def _generate_neighbor(self, cfg):
        new_cfg = copy.deepcopy(cfg)
        move = self.register.pick_move()
        move(new_cfg)
        return new_cfg

    def _acceptance_probability(self, diff, temperature):
        if diff <= 0:
            return 1.0
        return math.exp(- diff/ temperature)

    def _lower_temperature(self, temperature, iteration):
        t1 = self.end_temp + self.start_temp/(1 + math.log(iteration+1))
        t2 = self.start_temp + (self.end_temp - self.start_temp)*(iteration/self.max_iterations)
        return max(t1, t2)

    def _print_progress(
        self,
        iteration: int,
        best_solutions: List[List[str]],
        best_energies: List[float],
        current_solutions: List[List[str]],
        current_energies: List[float],
        temperature: float,
        start_time: float,
        spend_minute: int
    ) -> int:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed_time = time.time() - start_time

        # Check if 60 seconds have passed since the last update
        if elapsed_time - 60 * spend_minute > 60:
            spend_minute += 1
            progress = iteration / self.max_iterations * 100  # Progress as percentage

            # Print progress in a structured format
            print("===== Simulated Annealing Progress =====")
            print(f"Time: {current_time}")
            print(f"Iteration: {iteration}/{self.max_iterations} ({progress:.2f}%)")
            print(f"Temperature: {temperature:.4f}")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            # Print best solutions and energies
            print("\nBest Solutions:")
            for i, solution in enumerate(best_solutions):
                print(f"  Solution {i+1}: {solution}")
            print("\nBest Energies:")
            print("  " + ", ".join(f"{exp(energy):.4f}" for energy in best_energies))

            # Print current solutions and energies
            print("\nCurrent Solutions:")
            for i, solution in enumerate(current_solutions):
                print(f"  Solution {i+1}: {solution}")
            print("\nCurrent Energies:")
            print("  " + ", ".join(f"{exp(energy):.4f}" for energy in current_energies))

            print("========================================\n")

        return spend_minute

    def solve_batch(self, text_list):
        """
        Perform Simulated Annealing for multiple texts at once.
        """
        solutions = text_list[:]
        current_energies = self.cost_fn(solutions)

        best_solutions = solutions[:]
        best_energies = current_energies[:]

        log_energies = [[] for _ in range(len(text_list))]
        for i in range(len(text_list)):
            log_energies[i].append(current_energies[i])

        temperature = self.start_temp
        start_time = time.time()
        spend_minute = 0

        for iteration in range(self.max_iterations):
            # 1) Generate neighbors
            new_solutions = [self._generate_neighbor(sol) for sol in solutions]

            # 2) Calculate new energies in batch
            new_energies = self.cost_fn(new_solutions)

            # 3) Acceptance and update
            for i in range(len(text_list)):

                diff = new_energies[i] - current_energies[i]
                ap = self._acceptance_probability(diff, temperature)

                if random.random() < ap:
                    solutions[i] = new_solutions[i]
                    current_energies[i] = new_energies[i]

                if current_energies[i] < best_energies[i]:
                    best_solutions[i] = solutions[i]
                    best_energies[i] = current_energies[i]

            # 4) Lower temperature
            temperature = self._lower_temperature(temperature, iteration)

            # 5) Log current energies
            for i in range(len(text_list)):
                log_energies[i].append(current_energies[i])

            # 6) Print progress (extracted into separate method)
            spend_minute = self._print_progress(
                iteration,
                best_solutions,
                best_energies,
                solutions,
                current_energies,
                temperature,
                start_time,
                spend_minute
            )

            # 7) Early stop if temperature is below threshold
            if temperature <= self.end_temp:
                print("Reached the minimum temperature. Exiting.")
                break

        print(f"Execution time: {time.time() - start_time:.4f}s")

        # Convert best solutions back to strings
        return best_solutions, best_energies, log_energies
        
        
def compute_cost(cfg_list: List[Configuration], batch_size: int = 4) -> float:
    all_items = [cfg.flatten_encode() for cfg in cfg_list]
    return scorer.get_perplexity(all_items)
    
    
sa_params = {
    'start_temp': 0.012,         # Initial temperature
    'end_temp': 0.0012,          # Final temperature, decreasing linearly
    'max_iterations': 100000,    # Number of iterations (approximately 4 hours for 100,000 iterations)
    "cost_fn": compute_cost,
}

sa_optimizer = SimulatedAnnealing(**sa_params)

states = [copy.deepcopy(sample5_config) for i in range(1)]
best_solutions, best_energies, log_scores = sa_optimizer.solve_batch(states)

for score in log_scores:
    plt.plot(score)
    
plt.xlabel('sa_iteration')
plt.ylabel('score')
plt.legend()
plt.show()