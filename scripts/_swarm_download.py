"""Download Swarm MAGx_LR_1B (1 Hz B_NEC + flags + position) per satellite and day into npz files."""
import datetime as dt
from pathlib import Path

import numpy as np
from viresclient import SwarmRequest

ROOT = Path.home() / "datasets" / "SWARM"
start, days = dt.date(2023, 1, 1), 31
for sat in "ABC":
    out = ROOT / sat
    out.mkdir(parents=True, exist_ok=True)
    for k in range(days):
        day = start + dt.timedelta(days=k)
        f = out / f"{day.isoformat()}.npz"
        if f.exists():
            continue
        r = SwarmRequest()
        r.set_collection(f"SW_OPER_MAG{sat}_LR_1B")
        r.set_products(measurements=["B_NEC", "Flags_B"], sampling_step="PT1S")
        df = r.get_between(f"{day}T00:00:00Z", f"{day + dt.timedelta(days=1)}T00:00:00Z",
                           show_progress=False).as_dataframe(expand=True)
        np.savez_compressed(f, t=df.index.values.astype("datetime64[ns]").astype(np.int64),
                            B_NEC=df[["B_NEC_N", "B_NEC_E", "B_NEC_C"]].to_numpy(), flags=df["Flags_B"].to_numpy(),
                            lat=df["Latitude"].to_numpy(), lon=df["Longitude"].to_numpy(), radius=df["Radius"].to_numpy())
        print(sat, day, len(df), flush=True)
print("done")
