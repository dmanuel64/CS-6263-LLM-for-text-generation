'''
Large language models for code generation.
'''

import logging
import plotly.graph_objects as go
import sacrebleu
import torch

from bert_score import score as bertscore
from datasets import Dataset, load_dataset
from enum import Enum
from pathlib import Path
from peft.tuners.lora import LoraConfig
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import setup_chat_format
from trl.trainer import SFTTrainer
from typing import Literal, overload

logger = logging.getLogger('llmftg')


class SupportedModel(Enum):
    '''
    Type of model that can generate code.
    '''
    LLAMA = 'meta-llama/Llama-2-7b-hf', 'Llama 2'
    '''
    Llama 2-7B.
    '''
    PHI_2 = 'microsoft/phi-2', 'Phi-2'
    '''
    Phi-2.
    '''
    MISTRAL = 'mistralai/Mistral-7B-v0.1', 'Mistral'
    '''
    Mistral 7B
    '''

    def __init__(self, model_name: str, display_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        self._display_name = display_name

    @property
    def directory_name(self) -> str:
        '''
        Name of the directory where a fine-tuned model should be contained.

        Returns:
            The string name of the directory.
        '''
        return self._model_name.split('/')[-1]

    @property
    def display_name(self) -> str:
        '''
        The short-hand name of the model.

        Returns:
            The model's display name.
        '''
        return self._display_name

    @overload
    def get_hf(self, hf_attribute: Literal['model']) -> PreTrainedModel: ...

    @overload
    def get_hf(
        self, hf_attribute: Literal['tokenizer']) -> PreTrainedTokenizer: ...

    def get_hf(self, hf_attribute: Literal['model', 'tokenizer']) -> PreTrainedModel | PreTrainedTokenizer:
        '''
        Retrieves a model or tokenizer from Hugging Face.

        Parameters:
            hf_attribute: "model" or "tokenizer".

        Returns:
            A pre-trained model or pre-trained tokenizer.
        '''
        kwargs = {'trust_remote_code': True} if self is SupportedModel.PHI_2 else {}
        if hf_attribute == 'model':
            return AutoModelForCausalLM.from_pretrained(self._model_name, **kwargs, device_map='auto',
                                                        torch_dtype=torch.bfloat16)

        else:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name,
                                                      **kwargs)
            tokenizer.padding_side = 'right'
            return tokenizer  # type: ignore

    @classmethod
    def from_name(cls, name: str) -> 'SupportedModel':
        '''
        Creates a SupportedModel from it's model name.

        Returns:
            The supported model with the associated name.

        Raises:
            ValueError: If the name is not a supported model.
        '''
        for model in cls:
            if model.directory_name == name.split('/')[-1]:
                return model
        raise ValueError(f'Not a supported model: {name}')


class LLM:
    '''
    Large language model that can be fined-tuned on the flytech/python-codes-25k dataset.
    '''

    def __new__(cls, *args, **kwargs) -> 'LLM':
        if not hasattr(cls, 'DATASET'):
            logger.info('Creating test and train split')
            # Create training split and test split from the dataset
            dataset: Dataset = load_dataset('flytech/python-codes-25k',
                                            split='train')  # type: ignore
            cls.DATASET = dataset.train_test_split(test_size=20/len(dataset))
            cls.DATASET['train'] = cls.DATASET['train'].shuffle()
        return super().__new__(cls)

    def __init__(self, path: Path, model: SupportedModel) -> None:
        '''
        Creates a new LLM.

        Parameters:
            path: Path to store the LLM at after fine-tuning.
            model: Type of LLM to create.
        '''
        self._path = path
        self._model = model

    @property
    def model(self) -> str:
        '''
        The name of the model.

        Returns:
            The name of the model.
        '''
        return self._model.display_name

    def train(self, num_samples: int | None = None, batch_size: int = 3) -> None:
        '''
        Fine-tunes the LLM on the flytech/python-codes-25k dataset.

        Parameters:
            num_samples: Optional specific number of samples to train on.
            batch_size: Training batch size.
        '''
        if self._path.exists():
            raise ValueError(f'{self.model} is already fine-tuned')
        if not num_samples:
            samples = LLM.DATASET['train']
        elif num_samples <= 0 or num_samples > len(LLM.DATASET['train']):
            raise ValueError('num_samples must be greater than 0 and less than ' +
                             f'{len(LLM.DATASET['train']) + 1}')
        else:
            samples = Dataset.from_dict(LLM.DATASET['train'][:num_samples])
        torch.cuda.empty_cache()
        hf_model, tokenizer = setup_chat_format(self._model.get_hf('model'),
                                                self._model.get_hf('tokenizer'))
        trainer = SFTTrainer(model=hf_model,
                             tokenizer=tokenizer,
                             train_dataset=samples,
                             max_seq_length=3072,
                             peft_config=LoraConfig(
                                 lora_alpha=128,
                                 lora_dropout=0.05,
                                 r=256,
                                 bias='none',
                                 target_modules='all-linear',
                                 task_type='CAUSAL_LM'
                             ),
                             dataset_text_field='text',
                             args=TrainingArguments(
                                 output_dir=str(
                                     self._path.parent / f'.{self._model.display_name}_checkpoints'),
                                 per_device_train_batch_size=batch_size,
                                 gradient_checkpointing=True,
                             ))
        logger.info(f'Training {len(samples)} samples on {self.model}')
        trainer.train()  # type: ignore
        trainer.model.save_pretrained(self._path)

    def test(self, top_k: int, beam_size: int, temperature: float, num_samples: int | None = None) -> dict[Literal['BLEU', 'Rouge-L', 'BERTScore', 'CodeBLEU'], float]:
        '''
        Evaluates the LLM on the test split of the flytech/python-codes-25k dataset.

        Parameters:
            top_k: The top k most likely tokens considered when sampling.
            beam_size: Number of beams to use during beam search decoding.
            temperatuer: Percentage controlling the randomness of generating text.
            num_samples: Optional specific number of test samples to evaluate on.

        Returns:
            The dictionary containing the evaluation metrics of the model: BLEU, Rouge-L, BERTScore, and CodeBLEU.
        '''
        # Check parameters
        if top_k < 1:
            raise ValueError('top_k must be a positive integer')
        if beam_size < 1:
            raise ValueError('beam_size must be a positive integer')
        if temperature <= 0 or temperature > 1:
            raise ValueError(
                'temperature must be greater than 0.0 and less than 1.0')
        # Get test samples
        if not num_samples:
            samples = LLM.DATASET['test']
        elif num_samples <= 0 or num_samples > len(LLM.DATASET['test']):
            raise ValueError('num_samples must be greater than 0 and less than ' +
                             f'{len(LLM.DATASET['test']) + 1}')
        else:
            samples = Dataset.from_dict(LLM.DATASET['test'][:num_samples])
        # Evaluate samples
        logger.info(f'Evaluating {len(samples)} samples on {self.model}')
        pipe = pipeline('text-generation', str(self._path),
                        tokenizer=self._model.get_hf('tokenizer'),
                        trust_remote_code=True)
        outputs: list[str] = []
        targets: list[str] = []
        for sample in samples:
            logger.info(f'User: {sample["instruction"]}')  # type: ignore
            output: str = pipe(sample['instruction'])[
                0]['generated_text']  # type: ignore
            outputs.append(output)  # type: ignore
            logger.info(f'{self.model}: {output}')
            expected_output = '\n'.join(
                [sample['input'], sample['output']])  # type: ignore
            logger.debug(f'Ground Truth: {expected_output}')
            targets.append(expected_output)
        logger.info('Calculating metrics')
        return {
            'BLEU': sum(LLM.get_bleu_score(output, target) for output, target in zip(outputs, targets)) / len(outputs),
            'Rouge-L': sum(LLM.get_rouge_l_score(output, target) for output, target in zip(outputs, targets)) / len(outputs),
            'BERTScore': sum(LLM.get_bertscore(output, target) for output, target in zip(outputs, targets)) / len(outputs),
            'CodeBLEU': sum(LLM.get_code_bleu_score(output, target) for output, target in zip(outputs, targets)) / len(outputs)
        }

    def create_layers_image(self, path: Path) -> None:
        '''
        Creates images from layers, 4, 8, 16, and 32, showing the token probability distribution.

        Parameters:
            path: Path to the directory where the images should be stored.
        '''
        path.mkdir(parents=True, exist_ok=True)
        prompt = 'Write a "hello world" program in Python3.'
        pipe = pipeline('text-generation', str(self._path),
                        tokenizer=self._model.get_hf('tokenizer'),
                        trust_remote_code=True)
        model = pipe.model
        tokenizer = pipe.tokenizer

        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model(**inputs)
        
        # Specify the layers for which you want to print the probabilities
        layers = [1, 2, 4, 8]

        # Print the probabilities for each layer
        for layer_num in layers:
            layer_logits = outputs['logits'].squeeze()[layer_num]
            layer_probs = torch.softmax(layer_logits, dim=-1)

            # Convert token IDs to tokens
            tokens = tokenizer.convert_ids_to_tokens(range(len(layer_probs)))

            # Print the probabilities for each token
            print(f'Layer {layer_num} token probabilities:')
            token_probs = list((token, prob.detach().numpy()) for token, prob in zip(tokens, layer_probs) if not token.startswith('<'))
            token_probs.sort(key=lambda x: x[1], reverse=True)
            token_probs = token_probs[:20]
            for token, prob in token_probs:
                print(f"{token}: {prob.item()}")
            # Create a bar chart
            fig = go.Figure(data=[go.Bar(x=[i[0] for i in token_probs], y=[i[1] for i in token_probs])])
            fig.update_layout(title='Token Probabilities', xaxis_title='Token', yaxis_title='Probability')
            fig.write_image(path / f'layer_{layer_num}_probabilities.png')
            print()

    @staticmethod
    def get_bleu_score(sample: str, target: str) -> float:
        '''
        Calculates the BLEU score of a sample compared to a reference.

        Parameters:
            sample: Sample to calculate BLEU score of.
            target: Reference for BLEU scoring.

        Returns:
            The calculated BLEU score of the sample.
        '''
        return sacrebleu.sentence_bleu(sample, [target]).score  # type: ignore

    @staticmethod
    def get_rouge_l_score(sample: str, target: str) -> float:
        '''
        Calculates the Rouge-L score of a sample compared to a reference.

        Parameters:
            sample: Sample to calculate Rouge-L score of.
            target: Reference for Rouge-L scoring.

        Returns:
            The calculated Rouge-L score of the sample.
        '''
        return Rouge().get_scores([sample], [target], avg=True)['rouge-l']['f']

    @staticmethod
    def get_bertscore(sample: str, target: str) -> float:
        '''
        Calculates the BERTScore of a sample compared to a reference.

        Parameters:
            sample: Sample to calculate BERTScore of.
            target: Reference for BERTScoring.

        Returns:
            The calculated BERTScore of the sample.
        '''
        _, __, f1 = bertscore([sample], [target], lang="en")
        return f1.mean().item()  # type: ignore

    @staticmethod
    def get_code_bleu_score(sample: str, target: str) -> float:
        '''
        Calculates the CodeBLEU score of a sample compared to a reference.

        Parameters:
            sample: Sample to calculate CodeBLEU score of.
            target: Reference for CodeBLEU scoring.

        Returns:
            The calculated CodeBLEU score of the sample.
        '''
        return sacrebleu.corpus_bleu([sample], [[target]]).score

    @classmethod
    def from_pretrained(cls, path: Path) -> 'LLM':
        '''
        Creates a LLM from a fine-tuned model.

        Parameters:
            path: Path to the fine-tuned model.

        Returns:
            The fine-tuned LLM.
        '''
        return cls(path, SupportedModel.from_name(path.name))
