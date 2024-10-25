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
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--city', type=str, default="chicago", help='To set one city, use the format "chicago". '
                                                                    'To set two cities, use the format "[chicago,philadelphia]"')
    parser.add_argument('--time', type=str, default="2010", help='To set one time, use the format "2010" or "mid_history". '
                                                                 'To set two times, use the format "[2010,2012]" or "[mid_history,late_history]"')
    parser.add_argument('--dataset', type=str, default="FireWeatherIndex_Wildfire.csv", help='To set one dataset, use the format "FireWeatherIndex_Wildfire.csv". '
                                                                   'To set two datasets, use the format "[FireWeatherIndex_Wildfire.csv,heatindex.csv]"')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['gen_model'] = cmd_args.model if cmd_args.model is not None else args['models']['gen_model']
    args['datasets']['city'] = cmd_args.city if cmd_args.city is not None else args['datasets']['city']
    args['datasets']['time'] = cmd_args.time if cmd_args.time is not None else args['datasets']['time']
    args['datasets']['dataset'] = cmd_args.dataset if cmd_args.dataset is not None else args['datasets']['dataset']
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
