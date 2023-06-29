from .tasks.vision import reconstruction
from typer import Typer

app = Typer()
app.add_typer(reconstruction.app, name="reconstruction")
app()
