import sys
import pandas as pd
from pathlib import Path

source_path = Path("uploads/Stamp_Act-1.xlsx")
sheet_name = "Pathivu siragugal "

try:
    print(f"Trying to read sheet: '{sheet_name}'")
    df = pd.read_excel(str(source_path), sheet_name=sheet_name, dtype=str)
    print(f"Success! Shape: {df.shape}")
except Exception as e:
    print(f"Failed! Exception: {type(e).__name__} - {e}")
