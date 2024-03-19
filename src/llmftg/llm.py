import math

from datasets import Dataset, load_dataset
from enum import Enum
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from typing import Generator


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

    def get_trainer(self, dataset: Dataset, training_args: TrainingArguments | None = None) -> Trainer:
        if self is SupportedModel.PHI_2:
            return Trainer(model=AutoModelForCausalLM.from_pretrained(self._model_name,
                                                                      trust_remote_code=True),
                           tokenizer=AutoTokenizer.from_pretrained(self._model_name,
                                                                   trust_remote_code=True),
                           train_dataset=dataset, args=training_args)  # type: ignore
        return Trainer(model=AutoModelForCausalLM.from_pretrained(self._model_name),
                       tokenizer=AutoTokenizer.from_pretrained(
                           self._model_name),
                       train_dataset=dataset, args=training_args)  # type: ignore

    @classmethod
    def from_name(cls, name: str) -> 'SupportedModel':
        for model in cls:
            if model.directory_name == name.split('/')[-1]:
                return model
        raise ValueError(f'Not a supported model: {name}')


class LLM:

    def __new__(cls, *args, **kwargs) -> 'LLM':

        def generate_dataset(train: bool) -> Generator[dict[str, str], None, None]:
            nonlocal dataset, test_indicies
            for idx, sample in enumerate(dataset):
                if train and idx not in test_indicies:
                    yield sample
                elif not train and idx in test_indicies:
                    yield sample
        if not hasattr(cls, 'TRAIN_DATASET') or not hasattr(cls, 'TEST_DATASET'):
            dataset: Dataset = \
                load_dataset(
                    'flytech/python-codes-25k')['train']  # type: ignore
            test_indicies = list(range(0, len(dataset),
                                       math.ceil(len(dataset) / 20)))
            cls.TRAIN_DATASET: Dataset = Dataset.from_generator(
                generate_dataset, gen_kwargs={'train': True})  # type: ignore
            cls.TEST_DATASET: Dataset = Dataset.from_generator(
                generate_dataset, gen_kwargs={'train': False})  # type: ignore
            assert len(cls.TEST_DATASET) == 20, \
                'There should be 20 test samples, but ' + \
                f'{len(cls.TEST_DATASET)} exist'
            assert len(cls.TEST_DATASET) + len(cls.TRAIN_DATASET) == len(dataset), \
                f'Test split ({len(cls.TEST_DATASET)} samples) + train split ' + \
                f'({len(cls.TRAIN_DATASET)} samples) does not add up to ' + \
                f'{len(dataset)} samples'
        return super().__new__(cls)

    def __init__(self, path: Path, model: SupportedModel) -> None:
        self._path = path
        self._model = model

    @property
    def model(self) -> str:
        return self._model.display_name

    def train(self) -> None:
        trainer = self._model.get_trainer(self.TRAIN_DATASET)
        trainer.train()
        trainer.model.save_pretrained(self._path)

    @classmethod
    def from_pretrained(cls, path: Path) -> 'LLM':
        return cls(path, SupportedModel.from_name(path.name))
