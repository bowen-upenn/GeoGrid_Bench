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

import prompts
import utils

import logging
logging.getLogger("httpx").disabled = True


class QueryLLM:
    def __init__(self, args):
        self.args = args

        self.use_url = args['models']['use_url']
        if self.use_url:
            # Load the URL for internal OpenAI models
            assert re.search(r'gpt', self.args['models']['llm_model']) or 'o' in self.args['models']['llm_model']

            with open("api_tokens/model_url.txt", "r") as url_file:
                self.url = url_file.read().strip()
            with open("api_tokens/user_name.txt", "r") as user_name_file:
                self.user = user_name_file.read().strip()
        else:
            # OpenAI API
            if re.search(r'gpt', self.args['models']['llm_model']) is not None or 'o' in self.args['models']['llm_model']:
                with open("api_tokens/openai_key.txt", "r") as api_key_file:
                    self.api_key = api_key_file.read().strip()

                self.client = OpenAI(api_key=self.api_key)
                self.assistant = self.client.beta.assistants.create(
                    name="Data Analyzer",
                    instructions="You are a helpful assistant that analyze climate data using language, programming codes, and tabular data.",
                    tools=[{"type": "code_interpreter"}],
                    model=self.args['models']['llm'],
                )
                self.thread = None

            # Google Gemini API
            elif re.search(r'gemini', self.args['models']['llm_model']) is not None:
                with open(os.path.join(token_path, "gemini_key.txt"), "r") as genai_key_file:
                    self.genai_key = genai_key_file.read()
                self.client = genai.Client(api_key=self.genai_key)

            # Anthropic Claude API
            elif re.search(r'claude', self.args['models']['llm_model']) is not None:
                with open(os.path.join(token_path, "claude_key.txt"), "r") as claude_key_file:
                    self.claude_key = claude_key_file.read()
                self.client = anthropic.Client(api_key=self.claude_key)

            # Lambda Cloud API for LLaMA models
            else:
                with open(os.path.join(token_path, "lambda_key.txt"), "r") as lambda_key_file:
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
                    "max_tokens": 2000,
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
                if re.search(r'gpt', self.args['models']['llm_model']) is not None or 'o' in self.args['models']['llm_model']:
                    response = self.client.chat.completions.create(
                        model=self.args['models']['llm_model'],
                        messages=[{"role": "user",
                                   "content": prompt}],
                        max_tokens=2000
                    )
                    response = response.choices[0].message.content
                    if verbose:
                        print("model response: ", response)

                # Call Google Gemini API for Gemini models
                elif re.search(r'gemini', self.args['models']['llm_model']) is not None:
                    messages = [{"role": "user",
                                 "content": prompt}],
                    messages = utils.openai_to_gemini_history(messages)
                    response = self.client.models.generate_content(
                        model=self.args['models']['llm_model'],
                        contents=messages,
                        max_output_tokens=2000,
                    )
                    response = response.text

                # Call Claude API for Claude models
                elif re.search(r'claude', self.args['models']['llm_model']) is not None:
                    response = self.client.messages.create(
                        model=self.args['models']['llm_model'],
                        messages=[{"role": "user",
                                   "content": prompt}],
                        max_tokens=2000
                    )
                    response = response.content[0].text

                # Call lambda API for other models
                else:
                    model = self.args['models']['llm_model']
                    chat_completion = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user",
                                   "content": prompt}],
                        max_tokens=2000
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
