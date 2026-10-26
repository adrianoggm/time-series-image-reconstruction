# Sipers et al.: frequency threshold chosen on calibration windows (T = 128)

Windows outside every evaluated sample at T = 128. 8-bit NMAE, mean over channel series.

| Dataset | Windows | Criterion | tol 0.01 | tol 0.001 | tol 0.0001 | tol 1e-06 | tol 0 |
|---|---|---|---|---|---|---|---|
| wisdm | 100 | unsigned_oracle | 0.0403 | 0.0083 | 0.0026 | 0.0031 | 0.0039 |
| wisdm | 100 | signed | 0.0398 | 0.0087 | 0.0040 | 0.0057 | 0.0060 |
| pamap2 | 100 | unsigned_oracle | 0.0295 | 0.0077 | 0.0031 | 0.0038 | 0.0042 |
| pamap2 | 100 | signed | 0.0294 | 0.0081 | 0.0047 | 0.0061 | 0.0063 |
| esc50 | 100 | unsigned_oracle | 0.0350 | 0.0108 | 0.0040 | 0.0045 | 0.0052 |
| esc50 | 100 | signed | 0.0349 | 0.0107 | 0.0057 | 0.0071 | 0.0093 |
| ptbxl | 100 | unsigned_oracle | 0.0187 | 0.0046 | 0.0034 | 0.0054 | 0.0060 |
| ptbxl | 100 | signed | 0.0184 | 0.0059 | 0.0054 | 0.0074 | 0.0075 |
| swarm | 100 | unsigned_oracle | 0.0412 | 0.0382 | 0.0020 | 0.0019 | 0.0021 |
| swarm | 100 | signed | 0.0412 | 0.0377 | 0.0044 | 0.0052 | 0.0052 |
| stead | 100 | unsigned_oracle | 0.0244 | 0.0064 | 0.0035 | 0.0051 | 0.0057 |
| stead | 100 | signed | 0.0239 | 0.0070 | 0.0051 | 0.0072 | 0.0072 |
| **mean** | | unsigned_oracle | 0.0315 | 0.0127 | 0.0031 | 0.0040 | 0.0045 | chosen 0.0001 |
| **mean** | | signed | 0.0313 | 0.0130 | 0.0049 | 0.0065 | 0.0069 | chosen 0.0001 |
