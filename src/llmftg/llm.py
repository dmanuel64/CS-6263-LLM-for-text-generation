import logging
import torch

from datasets import Dataset, load_dataset
from enum import Enum
from pathlib import Path
from peft.tuners.lora import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import setup_chat_format
from trl.trainer import SFTTrainer
from typing import Literal, overload

logger = logging.getLogger('llmftg')


class SupportedModel(Enum):
    LLAMA = 'meta-llama/Llama-2-7b-hf', 'Llama 2'
    PHI_2 = 'microsoft/phi-2', 'Phi-2'
    MISTRAL = 'mistralai/Mistral-7B-v0.1', 'Mistral'

    def __init__(self, model_name: str, display_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        self._display_name = display_name

    @property
    def directory_name(self) -> str:
        return self._model_name.split('/')[-1]

    @property
    def display_name(self) -> str:
        return self._display_name

    @overload
    def get_hf(self, hf_attribute: Literal['model']) -> PreTrainedModel: ...

    @overload
    def get_hf(
        self, hf_attribute: Literal['tokenizer']) -> PreTrainedTokenizer: ...

    def get_hf(self, hf_attribute: Literal['model', 'tokenizer']) -> PreTrainedModel | PreTrainedTokenizer:
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
        for model in cls:
            if model.directory_name == name.split('/')[-1]:
                return model
        raise ValueError(f'Not a supported model: {name}')


class LLM:

    def __new__(cls, *args, **kwargs) -> 'LLM':
        def create_conversation(sample: dict[str, str]) -> dict[Literal['messages'], list[dict[Literal['role', 'content'], str]]]:
            return {
                'messages': [
                    {'role': 'system', 'content': 'You are a text to Python code translator. ' +
                     'Users will give you coding tasks in English and you will generate Python code based on the provided text.'},
                    {'role': 'user', 'content': sample['instruction']},
                    {'role': 'assistant', 'content': sample['output']}
                ]
            }
        if not hasattr(cls, 'DATASET'):
            logger.info('Creating test and train split')
            # Create training split and test split from the dataset
            dataset: Dataset = load_dataset('flytech/python-codes-25k',
                                            split='train')  # type: ignore
            # dataset.map(create_conversation, remove_columns=dataset.features,
            #             batched=False)
            cls.DATASET = dataset.train_test_split(test_size=20/len(dataset))
            cls.DATASET['train'] = cls.DATASET['train'].shuffle()
        return super().__new__(cls)

    def __init__(self, path: Path, model: SupportedModel) -> None:
        self._path = path
        self._model = model

    @property
    def model(self) -> str:
        return self._model.display_name

    def train(self) -> None:
        if self._path.exists():
            raise ValueError(f'{self.model} is already fine-tuned')
        torch.cuda.empty_cache()
        hf_model, tokenizer = setup_chat_format(self._model.get_hf('model'),
                                                self._model.get_hf('tokenizer'))
        # Fine-tuning LLMs source: https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
        trainer = SFTTrainer(model=hf_model,
                             tokenizer=tokenizer,
                             train_dataset=LLM.DATASET['train'],
                             max_seq_length=3072,
                             packing=True,
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
                                 num_train_epochs=3,
                                 per_device_train_batch_size=3,
                                 gradient_accumulation_steps=2,
                                 gradient_checkpointing=True,
                                 optim='adamw_torch_fused',
                                 logging_steps=10,
                                 save_strategy='epoch',
                                 learning_rate=2e-4,
                                 max_grad_norm=0.3,
                                 warmup_ratio=0.03,
                                 lr_scheduler_type='constant',
                             ))
        logger.info(f'Training {self.model}')
        trainer.train()
        trainer.model.save_pretrained(self._path)

    def test(self, top_k: int, beam_size: int, temperature: float, num_samples: int | None = None) -> dict[str, float]:
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
                        tokenizer=self._model.get_hf('tokenizer'))

    @classmethod
    def from_pretrained(cls, path: Path) -> 'LLM':
        return cls(path, SupportedModel.from_name(path.name))
