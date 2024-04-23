'''
Command line interface functionality.
'''

from click import command
from llmftg.llm import LLM, SupportedModel

from pathlib import Path
from rich import print
from rich.table import Table
from typer import Argument, BadParameter, Option, Typer
from typing import Annotated

import os
import llmftg
import shutil
import subprocess
import sys
import logging

logger = logging.getLogger('llmftg')
app = Typer()
'''
Main CLI app.
'''


def _toggle_logging(enabled: bool) -> None:
    logging.getLogger().setLevel(logging.INFO if enabled else logging.CRITICAL + 10)


@app.command()
def finetune(models: Annotated[Path, Argument(file_okay=False,
                                              help='Directory where the fine-tuned models are stored. ' +
                                              'If the directory does not exist, or --retrain is set, then ' +
                                              'the models will be trained first and saved in this ' +
                                              'directory.')],
             top_k: Annotated[int, Option(min=1,
                                          help='The top k most likely tokens considered when sampling. ' +
                                          'This parameter helps the models from generating unlikely or ' +
                                          'nonsensical tokens.')] = 20,
             beam_size: Annotated[int, Option(min=1,
                                              help='Number of beams to use during beam search decoding. ' +
                                              'A larger beam size can lead to more diverse, but potentially ' +
                                              'less fluent text.')] = 3,
             temperature: Annotated[float, Option(min=0.01, max=1.0,
                                                  help='Parameter controlling the randomness of generating text. ' +
                                                  'Lower temperatures produce more "safer" text, while higher ' +
                                                  'temperatures produce more creative, but potentially less coherent text.')] = 0.7,
             batch_size: Annotated[int, Option(min=1,
                                               help='Training batch size.')] = 3,
             train_samples: Annotated[int, Option(min=1, max=49606,
                                                  help='Number of samples to use during training.')] = 49606,
             test_samples: Annotated[int, Option(min=1, max=20,
                                                 help='Number of samples to use during evaluation.')] = 20,
             retrain: Annotated[bool, Option(
                 help="Delete the contents of fine-tuned models' directory and retrain all models.")] = False,
             verbose: Annotated[bool, Option(
                 help='Display verbose logging information.')] = False,
             accelerate: Annotated[bool, Option(help='Run with accelerate enabled')] = False) -> None:
    '''
    Fine-tunes LLama 2, Phi-2, and Mistral 7B on the flytech/python-codes-25k dataset, demonstrating how LLM
    hyperparameters such as top_k, beam_size, and temperature affect the generated output of the model.
    '''
    _toggle_logging(verbose)
    if accelerate:
        # Relaunch script in accelerate process
        logger.info('Relaunching script with accelerate')
        args = sys.argv[1:]
        args.remove('--accelerate')
        current_dir = Path.cwd()
        os.chdir(Path(llmftg.__file__).parent.parent)
        subprocess.run(['accelerate', 'launch', '--config_file',
                        Path(__file__).parent / 'resources' / 'accelerate_config.yaml', 'llmftg/__main__.py', *args])
        os.chdir(Path(current_dir))
    else:
        if retrain:
            # Delete fine-tuned models directory
            shutil.rmtree(models, ignore_errors=True)
        if not models.exists():
            models.mkdir(exist_ok=True, parents=True)
            # Fine-tune models on dataset
            for llm in (LLM(models / m.directory_name, m) for m in SupportedModel):
                try:
                    print(f'[yellow]Beginning to train {llm.model}...')
                    llm.train(num_samples=train_samples, batch_size=batch_size)
                except:
                    print(f'[red]Training failed for {llm.model}')
        items = list(i for i in models.glob('*') if not i.name.startswith('.'))
        if len(items) < len(SupportedModel):
            raise BadParameter(f'Expected {len(SupportedModel)} models in {models}. ' +
                               'Use --retrain to clear the directory',
                               param_hint='models')
        # Evaluate LLMs and collect metrics
        metrics = {m.model: m.test(top_k=top_k, beam_size=beam_size, temperature=temperature,
                                   num_samples=test_samples) for m in (LLM.from_pretrained(i) for i in items)}
        # Create table
        table = Table(title='Fine-Tuned Model Results')
        for column in ['Model', 'Top k', 'Beam Size', 'Temperature', 'BLEU', 'Rouge-L', 'BERTScore', 'CodeBLEU']:
            table.add_column(column)
        for model, model_metrics in metrics.items():
            table.add_row(model, str(top_k), str(beam_size), str(temperature),
                          f"{round(model_metrics['BLEU'], 3)}",
                          f"{round(model_metrics['Rouge-L'], 3)}",
                          f"{round(model_metrics['BERTScore'], 3)}",
                          f"{round(model_metrics['CodeBLEU'], 3)}")
        print(table)


@app.command()
def layers(model: Annotated[Path, Argument(file_okay=False,
                                           help='Directory where the fine-tuned model is stored.')],
           images: Annotated[Path, Argument(file_okay=False,
                                            help='Path to directory where the probability distribution images should be stored at.')]) -> None:
    '''
    Shows the probability distribution of each layer using a fine-tuned model.
    '''
    llm = LLM.from_pretrained(model)
    llm.create_layers_image(images)
