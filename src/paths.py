from pathlib import Path
import os

# Define the working directories so that they can be imported to our app
PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = PARENT_DIR / "data" / "raw"
TRANSFORMED_DATA_DIR = PARENT_DIR / "data" / "transformed"

# If the working directories do not exist, create them.
if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

