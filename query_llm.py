import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re
import math

from openai import OpenAI

import prompts
import utils


class QueryLLM:
    def __init__(self, args):
        self.args = args
        # Load the API key
        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()

        self.client = OpenAI(api_key=self.api_key)
        self.assistant = self.client.beta.assistants.create(
            name="Data Analyzer",
            instructions="You are a helpful assistant that analyze climate data using language, programming codes, and tabular data.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm'],
        )
        self.thread = None

    def create_a_thread(self):
        self.thread = self.client.beta.threads.create()

    def query_llm(self, step='extract_location', content="", assistant=False, verbose=False):
        if step == 'extract_location':
            assert assistant == False
            prompt = content
        else:
            raise ValueError(f'Invalid step: {step}')

        if not assistant:
            response = self.client.chat.completions.create(
                model=self.args['models']['llm'],
                messages=[{"role": "user",
                           "content": prompt}],
                max_tokens=300
            )
            response = response.choices[0].message.content
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
                print(run.status)

        return response
