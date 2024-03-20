'''
Command line interface functionality.
'''
import logging
import warnings

logger = logging.getLogger('llmftg')
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

from llmftg.llm import LLM, SupportedModel
from typer import Argument, BadParameter, Option, Typer
from typing import Annotated
from rich import print
from pathlib import Path
import torch
import shutil



app = Typer()
'''
Main CLI app.
'''


@app.command()
def command(models: Annotated[Path, Argument(file_okay=False,
                                             help='Directory where the fine-tuned models are stored. ' +
                                             'If the directory does not exist, or --retrain is set, then ' +
                                             'the models will be trained first and saved in this ' +
                                             'directory.')],
            top_k: Annotated[int, Option(min=1,
                                         help='The top k most likely tokens considered when sampling. ' +
                                         'This parameter helps the models from generating unlikely or ' +
                                         'nonsensical tokens.')] = 50,
            beam_size: Annotated[int, Option(min=1,
                                             help='Number of beams to use during beam search decoding. ' +
                                             'A larger beam size can lead to more diverse, but potentially ' +
                                             'less fluent text.')] = 3,
            temperature: Annotated[float, Option(min=0.01, max=1.0,
                                                 help='Parameter controlling the randomness of generating text. ' +
                                                 'Lower temperatures produce more "safer" text, while higher ' +
                                                 'temperatures produce more creative, but potentially less coherent text.')] = 0.7,
            test_samples: Annotated[int, Option(min=1, max=20,
                                                help='Number of test samples to use during evaluation.')] = 20,
            retrain: Annotated[bool, Option(
                help="Delete the contents of fine-tuned models' directory and retrain all models.")] = False,
            verbose: Annotated[bool, Option(help='Display verbose logging information.')] = False) -> None:
    if not torch.cuda.is_available():
        print('[red]You must have a GPU to run this command.')
    else:
        if verbose:
            logger.setLevel(logging.INFO)
        if retrain:
            shutil.rmtree(models, ignore_errors=True)
        if not models.exists():
            models.mkdir(exist_ok=True, parents=True)
            # Fine-tune models on dataset
            for llm in (LLM(models / m.directory_name, m) for m in [SupportedModel.PHI_2]):
                llm.train()
        items = list(i for i in models.glob('*') if not i.name.startswith('.'))
        if len(items) < len(SupportedModel):
            raise BadParameter(f'Expected {len(SupportedModel)} models in {models}. ' +
                               'Use --retrain to clear the directory',
                               param_hint='models')
        for llm in (LLM.from_pretrained(i) for i in items):
            llm.test(top_k=top_k, beam_size=beam_size, temperature=temperature)
