# Cost against window length (WISDM, one channel, single thread)

Median ms to encode + store 8 bits + decode; peak traced MB; exponent = log-log slope of time.

| Technique | L = 64 | L = 128 | L = 256 | L = 512 | L = 1024 | Exponent |
|---|---|---|---|---|---|---|
| Hirata · RP sin signo | 2.0 ms / 0 MB | 10.1 ms / 1 MB | 58.5 ms / 5 MB | 401.1 ms / 22 MB | 2871.2 ms / 87 MB | 2.64 |
| Hirata · Signed RP | 2.3 ms / 0 MB | 11.4 ms / 1 MB | 71.8 ms / 5 MB | 445.3 ms / 22 MB | 2880.1 ms / 87 MB | 2.59 |
| RP nueva · RP sin signo | 0.6 ms / 0 MB | 2.1 ms / 1 MB | 13.1 ms / 4 MB | 58.7 ms / 18 MB | 398.2 ms / 72 MB | 2.35 |
| RP nueva · Signed RP | 1.3 ms / 0 MB | 4.4 ms / 1 MB | 25.1 ms / 4 MB | 122.2 ms / 18 MB | 552.0 ms / 72 MB | 2.22 |
| GASF | 0.1 ms / 0 MB | 0.2 ms / 1 MB | 1.0 ms / 2 MB | 3.5 ms / 8 MB | 12.0 ms / 32 MB | 1.85 |
| GADF | 0.1 ms / 0 MB | 0.3 ms / 1 MB | 1.0 ms / 2 MB | 4.3 ms / 8 MB | 13.0 ms / 32 MB | 1.72 |
| MTF | 0.5 ms / 0 MB | 0.7 ms / 0 MB | 1.5 ms / 2 MB | 5.5 ms / 6 MB | 12.8 ms / 24 MB | 1.22 |
| XIRP | 0.1 ms / 0 MB | 0.2 ms / 1 MB | 0.6 ms / 2 MB | 4.1 ms / 8 MB | 12.8 ms / 32 MB | 2.03 |
| Sipers · RP sin signo (circular) | 1.1 ms / 1 MB | 7.3 ms / 2 MB | 27.1 ms / 9 MB | 184.6 ms / 31 MB | 1172.7 ms / 112 MB | 2.47 |
| Sipers · Signed RP (circular) | 1.5 ms / 1 MB | 5.9 ms / 2 MB | 29.9 ms / 9 MB | 202.5 ms / 30 MB | 1193.1 ms / 112 MB | 2.45 |
