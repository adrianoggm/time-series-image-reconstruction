import src.reconstruct_tseries as r

assert hasattr(r, "generate_and_save_time_series_fromGADF"), "GADF reconstruction not wired"
print("OK: generate_and_save_time_series_fromGADF is available")
