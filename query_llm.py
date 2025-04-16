import os
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './hf_home/hub'

from huggingface_hub import login
login("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")

import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re
import math

from openai import OpenAI
import requests
import anthropic
from google import genai  # Gemini has conflicting requirements of the environment with OpenAI
from google.genai.types import Part, UserContent, ModelContent
from transformers import AutoTokenizer, AutoModelForCausalLM

import prompts
import utils

import logging
logging.getLogger("httpx").disabled = True


class QueryLLM:
    def __init__(self, args, step='data_gen'):
        self.args = args
        self.step = step

        self.use_url = args['models']['use_url']
        if self.use_url:
            # Load the URL for internal OpenAI models
            assert re.search(r'gpt', self.args['models']['llm']) or \
                re.search(r'o1', self.args['models']['llm']) is not None or re.search(r'o3', self.args['models']['llm']) is not None

            with open("api_tokens/model_url.txt", "r") as url_file:
                self.url = url_file.read().strip()
            with open("api_tokens/user_name.txt", "r") as user_name_file:
                self.user = user_name_file.read().strip()
        else:
            # OpenAI API
            if re.search(r'gpt', self.args['models']['llm']) is not None or re.search(r'gpt', self.args['models']['llm']) is not None \
                    or re.search(r'o1', self.args['models']['llm']) is not None or re.search(r'o3', self.args['models']['llm']) is not None:
                with open("api_tokens/openai_key.txt", "r") as api_key_file:
                    self.api_key = api_key_file.read().strip()
                self.client = OpenAI(api_key=self.api_key)

                if self.step == 'data_gen':
                    self.assistant = self.client.beta.assistants.create(
                        name="Data Analyzer",
                        instructions="You are a helpful assistant that analyze climate data using language, programming codes, and tabular data.",
                        tools=[{"type": "code_interpreter"}],
                        model=self.args['models']['llm'],
                    )
                    self.thread = None

            # Google Gemini API
            elif re.search(r'gemini', self.args['models']['llm']) is not None:
                with open("api_tokens/gemini_key.txt", "r") as genai_key_file:
                    self.genai_key = genai_key_file.read()
                self.client = genai.Client(api_key=self.genai_key)

            # Anthropic Claude API
            elif re.search(r'claude', self.args['models']['llm']) is not None:
                with open("api_tokens/claude_key.txt", "r") as claude_key_file:
                    self.claude_key = claude_key_file.read()
                self.client = anthropic.Client(api_key=self.claude_key)

            # Hugging Face LLaMA models
            elif re.search(r'llama', self.args['models']['llm']) is not None:
                self.model_path = self.args['models']['llm']
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto')

            # Lambda Cloud API for LLaMA models
            else:
                with open("api_tokens/lambda_key.txt", "r") as lambda_key_file:
                    self.lambda_key = lambda_key_file.read()
                lambda_url = "https://api.lambdalabs.com/v1"
                self.client = OpenAI(api_key=self.lambda_key, base_url=lambda_url)

    def create_a_thread(self):
        self.thread = self.client.beta.threads.create()

    def query_llm(self, step='extract_location', content="", assistant=False, verbose=False):
        if step == 'extract_location':
            prompt = prompts.prompt_to_extract_location(content)
        elif step == 'rephrase_question':
            prompt = prompts.prompt_to_rephrase_question(content)
        elif step == 'filter_names':
            prompt = prompts.prompt_to_filter_names(content)
        elif step == 'inference':
            prompt = content
        else:
            raise ValueError(f'Invalid step: {step}')

        if not assistant:
            if self.use_url:
                # We use URL to access internal models only during inference, where we do not use the assistant API
                data = {
                    "user": self.user,
                    "model": self.args['models']['llm'],
                    "system": "You are a helpful data analyzer assistant.", # detailed instructions are provided in the prompt
                    "prompt": prompt,
                }

                # Convert the dict to JSON
                payload = json.dumps(data)
                headers = {'Content-Type': 'application/json'}

                # Send POST request
                response = requests.post(self.url, data=payload, headers=headers)

                # Receive the response data
                response = response.json()['response']

            else:
                # Call OpenAI API for GPT models by default
                if re.search(r'gpt', self.args['models']['llm']) is not None or re.search(r'gpt', self.args['models']['llm']) is not None \
                    or re.search(r'o1', self.args['models']['llm']) is not None or re.search(r'o3', self.args['models']['llm']) is not None:
                    response = self.client.chat.completions.create(
                        model=self.args['models']['llm'],
                        messages=[{"role": "user",
                                   "content": prompt}],
                    )
                    response = response.choices[0].message.content
                    if verbose:
                        print("model response: ", response)

                # Call Google Gemini API for Gemini models
                elif re.search(r'gemini', self.args['models']['llm']) is not None:
                    response = self.client.models.generate_content(
                        model=self.args['models']['llm'],
                        contents=prompt,
                    )
                    response = response.text

                # Call Claude API for Claude models
                elif re.search(r'claude', self.args['models']['llm']) is not None:
                    response = self.client.messages.create(
                        model=self.args['models']['llm'],
                        messages=prompt,
                        max_tokens=4096,
                    )
                    response = response.content[0].text

                # Use Hugging Face local LLaMA model
                elif re.search(r'llama', self.args['models']['llm']) is not None:
                    # Tokenize the prompt and generate response tokens using the local model
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    # Adjust max_new_tokens and other parameters as needed
                    outputs = self.model.generate(**inputs, max_new_tokens=4096)
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Call lambda API for other models
                else:
                    chat_completion = self.client.chat.completions.create(
                        model=self.args['models']['llm'],
                        messages=[{"role": "user",
                                   "content": prompt}],
                    )
                    response = chat_completion.choices[0].message.content

                if verbose:
                    print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
        else:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=prompt
            )

            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )

            if run.status == 'completed':
                response = self.client.beta.threads.messages.list(
                    thread_id=self.thread.id
                )
                response = response.data[0].content[0].text.value
                if verbose:
                    print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
            else:
                response = None

        return response
