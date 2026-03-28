# Basic information
This repo introduce a framework for video generation and understanding. 

# Usage
Run incontext subject-driven generation.

bash inference.sh

# Task
The code (univideo_inference.py) does not support multi-gpu inference. Please modify this python file and make it support multi-gpu inference. you may also use pytorch fsdp in your implementation. Output two files: univideo_inference_mgpu.py and inference_mgpu.sh