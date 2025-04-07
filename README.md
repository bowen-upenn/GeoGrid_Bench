## Dependencies
We use Python virtual environment. Please run the following commands to create a virtual environment and install all the requirements:
    
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

## Data Preparation
All benchmark data is available on [Google Drive](https://drive.google.com/drive/folders/1Kfs1YHHuTPIJt5OxZktBTtCc2sUIZWHA?usp=sharing). Please download the files and place them in the [data/benchmark/](data/benchmark/) directory. Make sure to unzip ```image_data.zip```. The file ```qa_data.csv``` contains all question–answer pairs in multiple-choice format, along with the corresponding image paths and other metadata for each question.

## Inference
Before you begin, create a new file named ```openai_key.txt``` in the root directory. Then, generate your OpenAI [API key](https://platform.openai.com/settings/organization/api-keys) and paste it into the ```openai_key.txt``` file.

We've provided inference scripts in the [scripts/](scripts/) directory. Please select the script based to the model name and run the following command as an example to run inference.

    bash scripts/run_inference_gpt4o.sh

You can choose from the following script files: ```run_inference_gpt4o.sh```, ```run_inference_gpt4o_mini.sh```, ```run_inference_o1.sh```, ```run_inference_o1_mini.sh```, ```run_inference_o3_mini.sh```, and ```run_inference_gpt-4.5-preview```. The price of each model is listed [here](https://platform.openai.com/docs/pricing). Inference results will be saved to the [result/](result/) directory automatically.



## Reference
- [ScienceAgentBench](https://arxiv.org/pdf/2410.05080)
