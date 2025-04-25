import os
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './hf_home/hub'

from huggingface_hub import login
login("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")

# spin up 16 threads (or however many cores you have)
os.environ["OMP_NUM_THREADS"]     = "16"
os.environ["MKL_NUM_THREADS"]     = "16"
os.environ["OPENBLAS_NUM_THREADS"]= "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re
import math
import io, contextlib, sys
from PIL import Image

from openai import OpenAI
import requests
# import anthropic
# from google import genai  # Gemini has conflicting requirements of the environment with OpenAI
# from google.genai.types import Part, UserContent, ModelContent
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, MllamaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

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
                if 'llama4' in self.model_path or 'vision' in self.model_path:
                    self.vision_model = MllamaForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto')
                self.processor = AutoProcessor.from_pretrained(self.model_path)

            elif re.search(r'Qwen', self.args['models']['llm']) is not None:
                self.model_path = self.args['models']['llm']
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype="auto", device_map="auto"
                )
                self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Lambda Cloud API for LLaMA models
            else:
                with open("api_tokens/lambda_key.txt", "r") as lambda_key_file:
                    self.lambda_key = lambda_key_file.read()
                lambda_url = "https://api.lambdalabs.com/v1"
                self.client = OpenAI(api_key=self.lambda_key, base_url=lambda_url)


    def create_a_thread(self):
        self.thread = self.client.beta.threads.create()


    def extract_code(self, response):
        """
        Extract the last Python code block in the response whose length > 10.
        If no such block exists, return the original response.
        """
        matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)

        # Filter and reverse scan to find the last valid one
        for code in reversed(matches):
            if len(code.strip()) > 10:
                return code.strip()

        return response


    def execute_code(self, response):
        """
        Execute a block of Python code and return its stdout output (including
        both the wrapper prints and any prints in the executed code), or
        an exception message if it fails.
        """
        response = self.extract_code(response)
        buffer = io.StringIO()

        # Redirect ALL prints into our buffer
        with contextlib.redirect_stdout(buffer):
            try:
                print('Executing code...')
                exec(response, {})
                print('Code executed successfully.')
            except Exception as e:
                # Any prints up to the error will be in buffer.getvalue()
                print('Code execution failed. Will continue with the original response.')
                return response, False

        # On success, buffer holds everything
        return buffer.getvalue(), True


    def query_llm(self, step='extract_location', content="", assistant=False, mode='text', verbose=False):
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
                    "system": "You are a helpful data analyzer assistant. The final answer should be one of (a), (b), (c), or (d).", # detailed instructions are provided in the prompt
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
                        max_tokens=2048,
                    )
                    response = response.content[0].text

                # Use Hugging Face local LLaMA model
                elif re.search(r'llama', self.args['models']['llm']) is not None:
                    # Tokenize the prompt and generate response tokens using the local model
                    if mode == 'image':
                        image = prompt['image']
                        text = prompt['text'] + '\nThe final answer should be one of (a), (b), (c), or (d).<|image|>'
                        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.model.device)
                    else:
                        text = prompt + '\nThe final answer should be one of (a), (b), (c), or (d).'
                        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

                    # Get eos token id safely
                    eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("</s>")
                    pad_token_id = self.tokenizer.pad_token_id or eos_token_id

                    # Adjust max_new_tokens and other parameters as needed
                    outputs = self.model.generate(**inputs, max_new_tokens=1024, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                elif re.search(r'Qwen', self.args['models']['llm']) is not None:
                    if mode == 'image':
                        text = self.processor.apply_chat_template(
                            prompt, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, _ = process_vision_info(prompt)
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = self.processor(
                            text=[text],
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")

                    generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    # print('output_text', output_text)


                # Call lambda API for other models
                else:
                    chat_completion = self.client.chat.completions.create(
                        model=self.args['models']['llm'],
                        messages=[{"role": "user",
                                   "content": prompt}],
                    )
                    response = chat_completion.choices[0].message.content

            # if mode == 'code':
            #     if verbose:
            #         print('!!!!!!!!!!!!!!init response', response)
            #     execution_result, success = self.execute_code(response)
            #     print('execution_result', execution_result)
            #     if success:
            #         response = execution_result
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
