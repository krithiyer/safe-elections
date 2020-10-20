# FAQ Answering System

## Installation
We reccommend creating a virtual environment first:
1. Install virtualenv if you haven't already:
`pip install virtualenv`
2. Create a virtual environment called faq:
`virtualenv faq`
4. Activate the environment:
`source faq/bin/activate`
. Install required packages:
`pip install -r requirements.txt`

Every time you start a new terminal, you need to activate this virtual environment (step 4) again.

## Downloading the paraphrase detection dataset
Run the following command:
`python download_dataset.py ./QQP`