# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
# BSD-3
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List

import ctranslate2
from transformers import T5Tokenizer
from huggingface_hub import snapshot_download
import numpy as np


class FastChat:

    def __init__(self, config):
        self.context_depth = config["context_depth"]
        self.max_tokens = config["max_tokens"]
        self.num_parallel_processes = config["num_parallel_processes"]
        self.num_threads_per_process = config["num_threads_per_process"]
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self) -> T5Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=self.tokenizer_model_name)
        return self._tokenizer

    @property
    def tokenizer_model_name(self) -> str:
        return "google/flan-t5-xl"

    @property
    def model(self) -> ctranslate2.Translator:
        if self._model is None:
            repo_path = snapshot_download(repo_id=self.llm_model_name)
            self._model = ctranslate2.Translator(model_path=repo_path,
                                                 intra_threads=self.num_threads_per_process,
                                                 inter_threads=self.num_parallel_processes)
        return self._model

    @property
    def llm_model_name(self) -> str:
        return "neongeckocom/fastchat-t5-3b-v1.0"

    @property
    def _system_prompt(self) -> str:
        return "A chat between a curious human and an artificial intelligence assistant. " \
               "The assistant gives helpful, detailed, and polite answers to the human's questions.\n" \
               "### Human: What are the key differences between renewable and non-renewable energy sources?\n" \
               "### Assistant: Renewable energy sources are those that can be " \
               "replenished naturally in a relatively short amount of time, such as solar, wind, hydro, " \
               "geothermal, and biomass. Non-renewable energy sources, on the other hand, " \
               "are finite and will eventually be depleted, such as coal, oil, and natural gas.\n"

    def ask(self, message: str, chat_history: List[List[str]]) -> str:
        """ Generates llm response based on user message and (user, llm) chat history """
        prompt = self._assemble_prompt(message, chat_history)
        llm_text_output = self._call_model(prompt)
        return llm_text_output

    def ppl(self, question: str, answers: List[str]) -> List[float]:
        """
            Computes PPL value for the list of provided answers
            :param question: Question for LLM to response to
            :param answers: List of provided answers
            :returns ppl values for each answer
        """
        question_prompt = self._assemble_prompt(question, [])
        log_probs_list = self._call_score(question_prompt, answers)
        ppl_list = [self._compute_ppl(log_probs) for log_probs in log_probs_list]
        return ppl_list

    @staticmethod
    def get_best(ppl_list: List[float]) -> int:
        """ Returns id of the minimal ppl value """
        best_id = np.argmin(ppl_list)
        return best_id

    def _call_model(self, prompt: str) -> str:
        """
            Wrapper for FastChat Model generation logic
            :param prompt: Input text sequence
            :returns: Output text sequence generated by model
        """
        tokens = self._tokenize(prompt)

        results = self.model.translate_batch(
            [tokens],
            beam_size=1,
            max_decoding_length=self.max_tokens,
            repetition_penalty=1.2,
        )

        output_tokens = results[0].hypotheses[0]
        text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(output_tokens),
                                     spaces_between_special_tokens=False)
        return text

    def _assemble_prompt(self, message: str, chat_history: List[List[str]]) -> str:
        """
            Assembles prompt engineering logic
            Setup Guidance:
            https://github.com/lm-sys/FastChat/blob/4e2c942b8d785eb5e2aef1d0df2150e756f381ab/fastchat/conversation.py#L279

            :param message: Incoming prompt
            :param chat_history: History of preceding conversation
            :returns: assembled prompt
        """
        prompt = self._system_prompt
        # Context N messages
        for role, content in chat_history[-self.context_depth:]:
            role_fastchat = self._convert_role(role)
            prompt += f"### {role_fastchat}: {content}\n"
        prompt += f"### Human: {message}\n### Assistant:"
        return prompt

    @staticmethod
    def _convert_role(role: str) -> str:
        """ Maps MQ role to FastChat internal domain """
        if role == "user":
            role_fastchat = "Human"
        elif role == "llm":
            role_fastchat = "Assistant"
        else:
            raise ValueError(f"role={role} is undefined, supported are: ('user', 'llm')")
        return role_fastchat

    def _call_score(self, prompt: str, targets: List[str]) -> List[List[float]]:
        """
            Calculates logarithmic probabilities for the list of provided text sequences
            :param prompt: Input text sequence
            :param targets: Output text sequences
            :returns: List of calculated logarithmic probabilities per output text sequence
        """
        tokens = self._tokenize(prompt)
        tokens_list = len(targets) * [tokens]

        target_tokens_list = [self._tokenize(target) for target in targets]

        results = self.model.score_batch(
            source=tokens_list,
            target=target_tokens_list,
        )

        log_probs_list = [result.log_probs for result in results]
        return log_probs_list

    def _tokenize(self, prompt: str) -> List[str]:
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))
        return tokens

    @staticmethod
    def _compute_ppl(log_probs: List[float]) -> float:
        """ Calculates perplexity value: https://en.wikipedia.org/wiki/Perplexity """
        ppl = np.exp(-np.mean(log_probs))
        return ppl
