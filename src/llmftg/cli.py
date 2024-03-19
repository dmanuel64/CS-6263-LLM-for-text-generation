'''
Command line interface functionality.
'''

from enum import Enum
from pathlib import Path
from typing import Annotated
from typer import Argument, Option, Typer

from llmftg.llm import LLM, SupportedModel

app = Typer()
'''
Main CLI app.
'''


@app.command()
def command(models: Annotated[Path, Argument(file_okay=False)]) -> None:
    print(LLM(models, SupportedModel.LLAMA))
