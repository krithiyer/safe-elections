# FAQ Answering System

## Installation
We reccommend creating a virtual environment first:
1. Install virtualenv if you haven't already:
`pip install virtualenv`
2. Create a virtual environment called faq:
`virtualenv faq`
4. Activate the environment:
`source faq/bin/activate`
5. Install required packages:
`pip install -r requirements.txt`

Every time you start a new terminal, you need to activate this virtual environment (step 4) again.

## Downloading the paraphrase detection dataset
Run the following command:
```
python download_dataset.py --data_dir ./QQP
```
You should see a folder named QQP/ with three files: train.tsv, dev.tsv and test.tsv. These are train, evaluation and test parts of the dataset respectively.


## Fine-tuning a pre-trained model on the paraphrase detection dataset

Run the follwoing command. After fine-tuning is done (which can take up to several on a GPU machine), the model weights will be saved in QQP/model/ folder.
```bash
python finetune.py --model_name_or_path bert-base-cased --task_name QQP --do_train --do_eval --evaluate_during_training --data_dir ./QQP --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./QQP/model/ --overwrite_output_dir
```

For the full list of available commandline options and a short description of what they do, run `python finetune.py --help`.
