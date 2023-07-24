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
from transformers import T5Tokenizer
from huggingface_hub import snapshot_download
import numpy as np


class FastChat:
    def __init__(self, config):
        self.model = config["model"]
        self.context_depth = config["context_depth"]
        self.max_tokens = config["max_tokens"]
        self.num_parallel_processes = config["num_parallel_processes"]
        self.num_threads_per_process = config["num_threads_per_process"]
        self.init_model()

    def init_model(self):
        repo_path = snapshot_download(repo_id="neongeckocom/fastchat-t5-3b-v1.0")
        self.model = ctranslate2.Translator(repo_path, 
                                            intra_threads=self.num_threads_per_process, 
                                            inter_threads = self.num_parallel_processes)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        self.system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: What are the key differences between renewable and non-renewable energy sources?\n### Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas.\n"

    @staticmethod
    def convert_role(role):
        if role == "user":
            role_fastchat = "Human"
        elif role == "llm":
            role_fastchat = "Assistant"
        return role_fastchat

    def assemble_prompt(self, message, chat_history):
        prompt = self.system_message
        # Context N messages
        for role, content in chat_history[-self.context_depth:]:
            role_fastchat = self.convert_role(role)
            prompt += f"### {role_fastchat}: {content}\n"
        prompt += f"### Human: {message}\n### Assistant:"
        return prompt

    def ask(self, message, chat_history):
        prompt = self.assemble_prompt(message, chat_history)
        bot_message = self.call_model(prompt)
        return bot_message

    def tokenize(self, prompt):
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))
        return tokens

    def call_model(self, prompt):
        tokens = self.tokenize(prompt)

        results = self.model.translate_batch(
            [tokens],
            beam_size=1,
            max_decoding_length = self.max_tokens,
            repetition_penalty = 1.2,
        )

        output_tokens = results[0].hypotheses[0]
        text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(output_tokens), spaces_between_special_tokens=False)
        return text

    def call_score(self, prompt, targets):
        tokens = self.tokenize(prompt)
        tokens_list = len(targets) * [tokens]

        target_tokens_list = [self.tokenize(target) for target in targets]

        results = self.model.score_batch(
            source = tokens_list,
            target = target_tokens_list,
        )

        log_probs_list = [result.log_probs for result in results]
        return log_probs_list

    @staticmethod
    def compute_ppl(log_probs):
        ppl = np.exp(-np.mean(log_probs))
        return ppl

    def ppl(self, question, answers):
        question_prompt = self.assemble_prompt(question, [])
        log_probs_list = self.call_score(question_prompt, answers)
        ppl_list = [self.compute_ppl(log_probs) for log_probs in log_probs_list]
        return ppl_list
    
    @staticmethod
    def get_best(ppl_list):
        best_id = np.argmin(ppl_list)
        return best_id