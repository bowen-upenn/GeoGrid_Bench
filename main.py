import numpy as np
import torch
import yaml
import os
import json
import argparse
import sys

from generate_dataset import generate_dataset


if __name__ == "__main__":
    print("Python", sys.version, 'Torch', torch.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--n', type=int, default=-1, help='Set number of samples to generate. Default is -1, which generates all samples')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm'] = cmd_args.llm if cmd_args.llm is not None else args['models']['llm']
    args['inference']['num_samples'] = cmd_args.n if cmd_args.n is not None else args['inference']['n']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Start inference
    print(args)
    generate_dataset(args)
