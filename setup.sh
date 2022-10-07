#!/bin/bash

wget https://github.com/sharvi24/event_detection/raw/main/maven_data.zip
unzip maven_data.zip

conda create --name event_detection
conda activate event_detection
conda install -c pytorch pytorch
conda install -c anaconda scikit-learn
pip install openprompt