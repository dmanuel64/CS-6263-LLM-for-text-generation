'''
Command line interface functionality.
'''
import shutil

from pathlib import Path
from typing import Annotated
from typer import Argument, BadParameter, Option, Typer

from llmftg.llm import LLM, SupportedModel

app = Typer()
'''
Main CLI app.
'''


@app.command()
def command(models: Annotated[Path, Argument(file_okay=False)],
            retrain: Annotated[bool, Option()] = False) -> None:
    if retrain:
        shutil.rmtree(models)
    if not models.exists():
        models.mkdir(exist_ok=True, parents=True)
        llms = [LLM(models / m.directory_name, m) for m in SupportedModel]
    else:
        llms = [LLM.from_pretrained(p) for p in models.glob('*')]
        if len(llms) < len(SupportedModel):
            raise BadParameter(f'Expected {len(SupportedModel)} models in {models}. ' +
                               'Use --retrain to clear the directory',
                               param_hint='models')
    print(llms)
