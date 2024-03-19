'''
Command line interface functionality.
'''

from typer import Argument, Option, Typer

app = Typer()
'''
Main CLI app.
'''

@app.command()
def command() -> None:
    pass