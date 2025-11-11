import pandas as pd
from pathlib import Path


ods_path = Path("data/kautian.ods")
out_dir = Path("data/csv_output")

sheets = pd.read_excel(ods_path, engine="odf", sheet_name=None)
out_dir.mkdir(exist_ok=True)

# iterate over each sheet and save to CSV
for sheet_name, df in sheets.items():
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in sheet_name)
    csv_path = out_dir / f"{safe_name}.csv"
    
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
