#!/bin/bash

set -e

cd "$(dirname "$0")/.."

# IF THE SCRIPT DOESN'T WORK, VISIT THIS WEBSITE FOR AN ALTERNATIVE COMMAND FOR INSTALLING FAISS
# https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
mamba install -y -c pytorch -c nvidia -c rapidsai -c conda-forge -c nvidia pytorch==2.5.1 pytorch-cuda=12.4 faiss-gpu
mamba install -y langchain sentence-transformers python-dotenv
pip install pandas langchain_huggingface google-genai
pip install -U langchain-community
pip install mysqlclient redis
pip install python-socketio aiohttp
