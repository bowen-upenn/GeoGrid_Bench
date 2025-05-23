## This is the official repository of the benchmark paper [GeoGrid-Bench: Can Foundation Models Understand Multimodal Gridded Geo-Spatial Data?](https://huggingface.co/datasets/bowen-upenn/GeoGrid_Bench)

[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/pdf/2505.10714)
[![Data](https://img.shields.io/badge/🤗HuggingFace-Link-FFA500)](https://huggingface.co/datasets/bowen-upenn/GeoGrid_Bench)

<p align="center">
<img src=image_formats.png/>
</p>


## 📊 Benchmark Data
We release the benchmark data of on [🤗Huggingface](https://huggingface.co/datasets/bowen-upenn/GeoGrid_Bench), including question-answer pairs, corresponding images, and other meta data. Please download the folder `image_data/` and the file `qa_data.csv`, and **put them under the ```data/benchmark/``` directory**.


## 🔗 Dependencies
We use Conda environment. Please run the following commands to create the environment and install all the requirements:
    
    conda create -n geospatial python=3.9
    conda activate geospatial
    pip install -r requirements.txt

## 🚀 Running Inference on Benchmark Data

### Step 1 - Credential Setup
**Before you begin**, create a new folder named [api_tokens/](api_tokens/) in the root directory. This folder will store your credentials required to run the models.


#### 🔸 If you are using **internal OpenAI models from Argonne National Laboratory**:
1. Inside the [api_tokens/](api_tokens/) folder, create the following two files:
- `model_url.txt` – containing the base URL of the internal OpenAI model endpoint
- `user_name.txt` – containing your Argonne username

#### 🔸 If you are running **open-sourced Llama models from Huggingface**:
1. Login with a [HF token](https://huggingface.co/settings/tokens) with gated access permission:
```bash 
huggingface-cli login
```
2. Request the model access from [HuggingFace](https://huggingface.co/meta-llama). 

#### 🔸 Otherwise, for **public API providers**, follow these steps:

1. **Create API keys** from the respective providers if you haven't already.

2. Inside the [api_tokens/](api_tokens/) folder, create the following text files and paste your API key as plain text into each:

- ```openai_key.txt``` – for OpenAI models
- ```gemini_key.txt``` – for Google Gemini models
- ```claude_key.txt``` – for Anthropic Claude models
- ```lambda_key.txt``` – for models accessed via the [Lambda Cloud API](https://docs.lambda.ai/public-cloud/lambda-inference-api/?_gl=1*1yqhedk*_gcl_aw*R0NMLjE3NDQwOTAyNjIuQ2owS0NRanc3ODJfQmhEakFSSXNBQlR2X0pEWUpQRTRhLXJMY0xjeWZYYUZrRzE4Q196MG0zdjY0cmQtX09FYm5iRHlrek95QWVDVVZxVWFBbnhYRUFMd193Y0I.*_gcl_au*NTQ3OTExNDIzLjE3NDQwOTAyNjE.*_ga*MTA0MjYwNjUyMS4xNzQ0MDkwMjYy*_ga_43EZT1FM6Q*MTc0NDA5MDI2MS4xLjAuMTc0NDA5MDI2MS42MC4wLjY1NjAyNzc2NA..) (e.g., Llama, etc.)


### Step 2 - Running Inference Scripts
We provide ready-to-use **inference scripts** in the [scripts/](scripts/) directory for evaluating the following models:
- **[OpenAI Models](https://platform.openai.com/docs/models)**
  - GPT-4.5: ```run_inference_gpt_4p5_preview.sh```
  - o3-mini: ```run_inference_o3_mini.sh```
  - o1: ```run_inference_o1.sh```
  - GPT-4o: ```run_inference_gpt_4o.sh```
  - GPT-4o-mini: ```run_inference_gpt_4o_mini.sh```
- **[Google Gemini Models](https://ai.google.dev/gemini-api/docs/models)**
  - Gemini-2.0-Flash: ```run_inference_gemini_2p0_flash.sh```
  - Gemini-1.5-Flash: ```run_inference_gemini_1p5_flash.sh```
- **[Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models/all-models)**
  - Claude-3.7-Sonnet: ```run_inference_claude_3p7_sonnet.sh```
  - Claude-3.5-Haiku: ```run_inference_claude_3p5_haiku.sh```
- **[Meta Llama Models](https://docs.lambda.ai/public-cloud/lambda-inference-api/?_gl=1*1yqhedk*_gcl_aw*R0NMLjE3NDQwOTAyNjIuQ2owS0NRanc3ODJfQmhEakFSSXNBQlR2X0pEWUpQRTRhLXJMY0xjeWZYYUZrRzE4Q196MG0zdjY0cmQtX09FYm5iRHlrek95QWVDVVZxVWFBbnhYRUFMd193Y0I.*_gcl_au*NTQ3OTExNDIzLjE3NDQwOTAyNjE.*_ga*MTA0MjYwNjUyMS4xNzQ0MDkwMjYy*_ga_43EZT1FM6Q*MTc0NDA5MDI2MS4xLjAuMTc0NDA5MDI2MS42MC4wLjY1NjAyNzc2NA..)**
  - Llama-4-Maverick: ```run_inference_llama4_maverick.sh```
  - Llama-4-Scout: ```run_inference_llama4_scout.sh```
  - Llama-3.3-70B: ```run_inference_llama3p3_70b.sh```
  - Llama-3.2-90B-Vision: ```run_inference_llama_3p2_90b_vision.sh```
  - Llama-3.2-11B-Vision: ```run_inference_llama_3p2_11b_vision.sh```
  - Llama-3.2-3B: ```run_inference_llama_3p2_3b.sh```
  - Llama-3.1-405B: ```run_inference_llama_3p1_405b.sh```

 🔮 **To run evaluation for a specific model, simply execute the corresponding script. For example:**
```bash
bash scripts/run_inference_gpt4o.sh [MODALITY] [START_IDX] [END_IDX]
```

- **[MODALITY]** can be one of the following: ```text```, ```code```, or ```image```.
> 💡Note that the ```image``` modality is only available for GPT-4o, GPT-4o-mini, o1, GPT-4.5-Preview, Claude-3.5-Haiku, Claude-3.7-Sonnet, Gemini-2.0-Flash, Gemini-1.5-Flash, Llama-4-Maverick, Llama-4-Scout, Llama-3.2-90B-Vision, and Llama-3.2-11B-Vision.

- **[START_IDX]** and **[END_IDX]** define the range of question indices for inference. The script will run inference starting at [START_IDX] and ending just before [END_IDX] (non-inclusive). 
> 💡Note that whenever you set [END_IDX] to -1, the script will run inference from [START_IDX] until the end of the dataset. Meanwhile, if you set [START_IDX] to 0, the script will clean up the exsiting file and start clean. Otherwise, the script will append new evaluation results to the existing result file.

- If you are using internal OpenAI models accessed by an URL, add `use_url` to the command line. For example:
```bash
bash scripts/run_inference_gpt4o.sh text 0 -1 use_url
```

**We provide a complete [checklist](scripts/checklist.txt) of all scripts used in our benchmark.**

### Step 3 - Saving Inference Results

**Evaluation results** will be automatically saved in the [result/](result/) directory, with filenames that include both the model name and the data modality for easy identification. For example, ```eval_results_gpt-4o_text.json```.
