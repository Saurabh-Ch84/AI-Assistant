import os
from dotenv import load_dotenv

load_dotenv()

START_SEQUENCE = "\033[90m"  # ANSI escape code for bright black / grey
END_SEQUENCE = "\033[0m"    # ANSI escape code to reset formatting

def log(text: str) -> None:
    """Prints debug statements in grey if DEBUG environment variable is set."""
    debug_mode = str(os.getenv("DEBUG", "")).strip().lower() in ("true", "1", "yes")
    if debug_mode:
        print(f"  {START_SEQUENCE}{text}{END_SEQUENCE}", flush=True)