from .tasks.vision import pixel_reconstruction
from typer import Typer

app = Typer(pretty_exceptions_enable=False)
app.add_typer(pixel_reconstruction.app, name="pixel-reconstruction")
app()
