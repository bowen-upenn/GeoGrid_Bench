import os
import numpy as np
import torch
import json

from data_retrieval import retrieve_data_from_location
from query_llm import QueryLLM
import utils


def generate_dataset(args):
    cities, times, datasets = utils.parse_inputs(args)
    print(f"Generating dataset samples for {cities} at {times} with database {datasets}")

    llm = QueryLLM(args)

    for city in cities:
        for time in times:
            for dataset in datasets:
                data = retrieve_data_from_location(dataset, city, time, llm)

