# LLMs for Code Generation
A tool demonstrating how to use LLMs for code generation using Hugging Face and PyTorch. This tool fine-tunes Llama2, Phi-2, and Mistral LLMs on a train split of the [flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) dataset, where 20 samples will be the test split. The tool displays various metrics regarding the models' generated code quality, and allows the user to tune hyperparameters to see how each hyperparameter affects the quality of the output. This tool additionally generates bar charts for the token probabilities of specific layers of the fine-tuned models.

Assignments 1.3 and 2 for CS 6263: Natural Language Processing

## Quick Start
Install with `pip`:
```
pip install git+https://github.com/dmanuel64/LLM-for-text-generation.git#egg=llmftg
```
Specify a path to store/load the fine-tuned models:
```
python -m llmftg finetune ~/fine_tuned_models
```
To run with `accelerate` (potentially faster training time,) run with `--accelerate`:
```
python -m llmftg finetune ~/fine_tuned_models --accelerate
```
To see the token probabilities of each layer from a specific model, run:
```
python -m llmftg layers ~/fine_tuned_models/phi-2 layer_images
```

## Usage
For general usage information:
```
python -m llmftg --help
```