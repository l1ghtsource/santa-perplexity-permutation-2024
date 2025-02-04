import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch

from itertools import permutations

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = '/kaggle/input/gemma-2/transformers/gemma-2-9b/2',
    load_in_8bit: bool = False,
    clear_mem: bool = False,
) -> float:
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, 'text'].str.split().apply(Counter)
    sub_counts = submission.loc[:, 'text'].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            'At least one submitted string is not a valid permutation of the solution string.'
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [
        ' '.join(s.split()) for s in submission['text'].tolist()
    ]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(
        sub_strings
    )  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print('GPU memory clearing failed.')

    return float(np.mean(perplexities))


class PerplexityCalculator:

    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device_map: str = 'auto',
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires CUDA device')
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        self.model.eval()

    def get_perplexity(
        self, input_texts: Union[str, List[str]], debug=False
    ) -> Union[float, List[float]]:
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        with torch.no_grad():
            # Process each sequence independently
            for text in input_texts:
                # Explicitly add sequence boundary tokens to the text
                text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"

                # Tokenize
                model_inputs = self.tokenizer(
                    text_with_special,
                    return_tensors='pt',
                    add_special_tokens=False,
                )

                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                logits = output['logits']

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = model_inputs['input_ids'][..., 1:].contiguous()  # Drop first input

                # Calculate token-wise loss
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Calculate average loss
                sequence_loss = loss.sum() / len(loss)
                loss_list.append(sequence_loss.cpu().item())

                # Debug output
                if debug:
                    print(f"\nProcessing: '{text}'")
                    print(f"With special tokens: '{text_with_special}'")
                    print(f"Input tokens: {model_inputs['input_ids'][0].tolist()}")
                    print(f"Target tokens: {shift_labels[0].tolist()}")
                    print(f"Input decoded: {self.tokenizer.decode(model_inputs['input_ids'][0])}")
                    print(f"Target decoded: {self.tokenizer.decode(shift_labels[0])}")
                    print(f"Individual losses: {loss.tolist()}")
                    print(f"Average loss: {sequence_loss.item():.4f}")

        ppl = [exp(i) for i in loss_list]

        if debug:
            print("\nFinal perplexities:")
            for text, perp in zip(input_texts, ppl):
                print(f"Text: '{text}'")
                print(f"Perplexity: {perp:.2f}")

        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            
            
df = pd.read_csv("/kaggle/input/santa-249-09/new (3) (1).csv")
scorer = PerplexityCalculator("/kaggle/input/gemma-2/transformers/gemma-2-9b/2")


def find_best_pairwise_permutation(input_str):
    words = input_str.split()

    best_perplexity = scorer.get_perplexity(input_str)
    print(f"start perp: {best_perplexity}, for sentence <{input_str}>")
    best_sentence = input_str

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            swapped_words = words[:]
            swapped_words[i], swapped_words[j] = swapped_words[j], swapped_words[i]
            swapped_sentence = ' '.join(swapped_words)

            perplexity = scorer.get_perplexity(swapped_sentence)
            if perplexity < best_perplexity:
                find_best_pairwise_permutation(swapped_sentence)
        print(i)
        
text = df.loc[4, 'text']
find_best_pairwise_permutation(text)

processed_sentences = set()


def find_best_reorder(input_str, index):
    words = input_str.split()
    if index < 0 or index >= len(words):
        raise ValueError("Index is out of range")

    best_sentence = input_str
    best_perplexity = scorer.get_perplexity(input_str)

    word_to_move = words.pop(index)

    for i in range(len(words) + 1):
        reordered_words = words[:]
        reordered_words.insert(i, word_to_move)
        reordered_sentence = ' '.join(reordered_words)

        perplexity = scorer.get_perplexity(reordered_sentence)

        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_sentence = reordered_sentence
            print(f"Found better permutation: <{reordered_sentence}> with perplexity: {perplexity}")

    return best_sentence if best_sentence != input_str else None


def optimize_sentence(input_str):
    global processed_sentences

    if input_str in processed_sentences:
        return
    processed_sentences.add(input_str)

    best_sentence = input_str
    best_perplexity = scorer.get_perplexity(input_str)
    print(f"Processing sentence: <{input_str}> with perplexity: {best_perplexity}")

    words = input_str.split()

    for index in range(len(words)):
        print(index)
        new_sentence = find_best_reorder(best_sentence, index)
        if new_sentence and new_sentence not in processed_sentences:
            optimize_sentence(new_sentence)
            
optimize_sentence(text)


def generate_permuted_sentences(text, num_words):
    words = text.split()
    all_sentences = []

    for i in range(len(words) - num_words + 1):
        prefix = words[:i]
        subarray = words[i:i + num_words]
        suffix = words[i + num_words:]
        
        for permuted in permutations(subarray):
            sentence = prefix + list(permuted) + suffix
            all_sentences.append(" ".join(sentence))
    
    return set(all_sentences)


def evaluate_perplexity(sentences):
    min_perplexity = float("inf")
    best_sentence = None

    for idx, sentence in enumerate(sentences, start=1):
        perplexity = scorer.get_perplexity(sentence)
        
        if perplexity < min_perplexity:
            min_perplexity = perplexity
            best_sentence = sentence
            print(f"Processed {idx} sentences. Current best perplexity: {min_perplexity}, <{sentence}>")
            
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(sentences)} sentences.")
    
    print(f"\nFinal best sentence: {best_sentence}")
    print(f"Final best perplexity: {min_perplexity}")
    
num_words = 4
sentences = generate_permuted_sentences(text, num_words)
print(len(sentences))

evaluate_perplexity(sentences)

