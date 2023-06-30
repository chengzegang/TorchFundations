from .tasks.vision import autoencoder
from typer import Typer

app = Typer(pretty_exceptions_enable=False)
app.add_typer(autoencoder.app, name="autoencoder")
app()
