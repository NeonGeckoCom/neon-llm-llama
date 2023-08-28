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
import ctranslate2
import numpy as np

from typing import List
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
from neon_llm_core.llm import NeonLLM


class Llama(NeonLLM):

    mq_to_llm_role = {
        "user": "[/INST]",
        "llm": "</s><s>[INST]"
    }

    def __init__(self, config):
        super().__init__(config)
        self.context_depth = config["context_depth"]
        self.max_tokens = config["max_tokens"]
        self.num_parallel_processes = config["num_parallel_processes"]
        self.num_threads_per_process = config["num_threads_per_process"]
        self.warmup()

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_pretrained(identifier=self.tokenizer_model_name)
        return self._tokenizer

    @property
    def tokenizer_model_name(self) -> str:
        return "neongeckocom/Llama-2-7b-chat-hf"

    @property
    def model(self) -> ctranslate2.Generator:
        if self._model is None:
            repo_path = snapshot_download(repo_id=self.llm_model_name)
            self._model = ctranslate2.Generator(model_path=repo_path,
                                                 device="auto",
                                                 intra_threads=self.num_threads_per_process,
                                                 inter_threads=self.num_parallel_processes)
        return self._model

    @property
    def llm_model_name(self) -> str:
        return "neongeckocom/Llama-2-7b-chat-hf"

    @property
    def _system_prompt(self) -> str:
        return ("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n")

    def warmup(self):
        self.tokenizer
        self.model

    def get_sorted_answer_indexes(self, question: str, answers: List[str]) -> List[int]:
        """
            Creates sorted list of answer indexes with respect to order provided in :param answers based on PPL score
            Answers are sorted from best to worst
            :param question: incoming question
            :param answers: list of answers to rank
            :returns list of indexes
        """
        if not answers:
            return []
        scores = self._ppl(question=question, answers=answers)
        sorted_items = sorted(zip(range(len(answers)), scores), key=lambda x: x[1])
        sorted_items_indexes = [x[0] for x in sorted_items]
        return sorted_items_indexes

    def _call_model(self, prompt: str) -> str:
        """
            Wrapper for Llama Model generation logic
            :param prompt: Input text sequence
            :returns: Output text sequence generated by model
        """
        tokens = self._tokenize(prompt)

        results = self.model.generate_batch(
            [tokens],
            include_prompt_in_result=False,
            max_length=self.max_tokens,
            repetition_penalty=1.2,
        )

        output_tokens = results[0].sequences_ids[0]
        text = self.tokenizer.decode(output_tokens)
        text = self._clean_responce(text)
        return text

    @staticmethod
    def _clean_responce(text: str) -> str:
        clean_text = text.strip()
        return clean_text

    def _assemble_prompt(self, message: str, chat_history: List[List[str]]) -> str:
        """
            Assembles prompt engineering logic
            Setup Guidance:
            https://github.com/lm-sys/FastChat/blob/4e2c942b8d785eb5e2aef1d0df2150e756f381ab/fastchat/conversation.py#L848

            :param message: Incoming prompt
            :param chat_history: History of preceding conversation
            :returns: assembled prompt
        """
        prompt = self._system_prompt
        # Context N messages
        for role, content in chat_history[-self.context_depth:]:
            role_llama = self.convert_role(role)
            prompt += f"{content} {role_llama} "
        prompt += f"{message} {self.mq_to_llm_role['user']}"
        return prompt

    def _call_score(self, prompt: str, targets: List[str]) -> List[List[float]]:
        """
            Calculates logarithmic probabilities for the list of provided text sequences
            :param prompt: Input text sequence
            :param targets: Output text sequences
            :returns: List of calculated logarithmic probabilities per output text sequence
        """
        prompt_len = len(self._tokenize(prompt))

        tokens_list = [
            self._tokenize(f"{prompt} {target}</s>")
            for target in targets
        ]

        results = self.model.score_batch(
            tokens=tokens_list
        )

        log_probs_list = [result.log_probs[prompt_len-1:] for result in results]
        return log_probs_list

    def _tokenize(self, prompt: str) -> List[str]:
        tokens = self.tokenizer.encode(prompt).tokens
        return tokens

    def _ppl(self, question: str, answers: List[str]) -> List[float]:
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
    def _compute_ppl(log_probs: List[float]) -> float:
        """ Calculates perplexity value: https://en.wikipedia.org/wiki/Perplexity """
        ppl = np.exp(-np.mean(log_probs))
        return ppl
