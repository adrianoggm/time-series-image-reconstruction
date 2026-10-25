# M7 classification: classifiers trained on original windows, tested on reconstructions

Accuracy for multiclass data, macro-F1 for multilabel PTB-XL. Grouped 3-fold, 8-bit images. Mark ✓: per-group accuracy difference within ±0.01 (95% cluster bootstrap).

## L = 512 · KNN

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.158 | 0.119 | 0.127 | 0.143 | 0.159 | 0.157 ✓ | 0.157 ✓ | 0.122 | 0.158 ✓ | 0.144 | 0.158 ✓ |
| esc10_4k | 0.163 | 0.127 | 0.132 | 0.149 | 0.163 ✓ | 0.163 ✓ | 0.163 ✓ | 0.159 | 0.163 ✓ | 0.151 | 0.162 ✓ |
| esc10_8k | 0.163 | 0.139 | 0.143 | 0.148 | 0.162 ✓ | 0.163 ✓ | 0.162 ✓ | 0.179 | 0.163 ✓ | 0.161 | 0.164 ✓ |

## L = 512 · SVM

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.260 | 0.250 | 0.240 | 0.262 | 0.260 ✓ | 0.261 ✓ | 0.260 ✓ | 0.246 | 0.260 ✓ | 0.263 | 0.260 ✓ |
| esc10_4k | 0.250 | 0.236 | 0.235 | 0.250 | 0.250 ✓ | 0.250 ✓ | 0.250 ✓ | 0.221 | 0.250 ✓ | 0.260 | 0.252 ✓ |
| esc10_8k | 0.236 | 0.228 | 0.216 | 0.248 | 0.237 ✓ | 0.237 ✓ | 0.237 ✓ | 0.249 | 0.236 ✓ | 0.251 | 0.235 ✓ |

## L = 512 · RF

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.319 | 0.245 | 0.253 | 0.296 | 0.326 ✓ | 0.323 ✓ | 0.321 ✓ | 0.200 | 0.322 ✓ | 0.290 | 0.326 |
| esc10_4k | 0.313 | 0.253 | 0.259 | 0.281 | 0.311 ✓ | 0.309 ✓ | 0.310 | 0.202 | 0.314 ✓ | 0.275 | 0.313 |
| esc10_8k | 0.308 | 0.282 | 0.282 | 0.309 | 0.308 ✓ | 0.314 | 0.311 ✓ | 0.252 | 0.308 ✓ | 0.307 | 0.309 ✓ |

## L = 512 · LSTM

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.260 | 0.199 | 0.200 | 0.236 | 0.260 ✓ | 0.259 ✓ | 0.258 ✓ | 0.190 | 0.260 ✓ | 0.231 | 0.259 ✓ |
| esc10_4k | 0.298 | 0.228 | 0.246 | 0.267 | 0.297 ✓ | 0.298 ✓ | 0.297 ✓ | 0.216 | 0.298 ✓ | 0.264 | 0.296 ✓ |
| esc10_8k | 0.274 | 0.233 | 0.239 | 0.255 | 0.275 ✓ | 0.274 ✓ | 0.274 ✓ | 0.234 | 0.274 ✓ | 0.253 | 0.271 ✓ |

## L = 512 · GRU

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.358 | 0.284 | 0.285 | 0.342 | 0.358 ✓ | 0.357 ✓ | 0.359 ✓ | 0.240 | 0.358 ✓ | 0.336 | 0.359 ✓ |
| esc10_4k | 0.395 | 0.320 | 0.325 | 0.377 | 0.395 ✓ | 0.395 ✓ | 0.394 ✓ | 0.280 | 0.395 ✓ | 0.374 | 0.394 ✓ |
| esc10_8k | 0.382 | 0.322 | 0.338 | 0.355 | 0.382 ✓ | 0.380 ✓ | 0.382 ✓ | 0.313 | 0.381 ✓ | 0.355 | 0.381 ✓ |

## L = 512 · CNN

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.467 | 0.330 | 0.348 | 0.429 | 0.467 ✓ | 0.467 ✓ | 0.469 ✓ | 0.314 | 0.467 ✓ | 0.429 | 0.465 ✓ |
| esc10_4k | 0.395 | 0.317 | 0.323 | 0.368 | 0.395 ✓ | 0.395 ✓ | 0.396 ✓ | 0.298 | 0.395 ✓ | 0.369 | 0.394 ✓ |
| esc10_8k | 0.459 | 0.350 | 0.377 | 0.433 | 0.458 ✓ | 0.460 ✓ | 0.460 ✓ | 0.383 | 0.459 ✓ | 0.427 | 0.462 ✓ |

## L = 1024 · KNN

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.123 | 0.107 | 0.115 | 0.125 | 0.123 ✓ | 0.123 ✓ | 0.125 ✓ | 0.113 | 0.123 ✓ | 0.123 | 0.124 ✓ |
| esc10_4k | 0.126 | 0.098 | 0.107 | 0.106 | 0.126 ✓ | 0.127 ✓ | 0.128 ✓ | 0.132 | 0.125 ✓ | 0.109 | 0.129 |
| esc10_8k | 0.160 | 0.126 | 0.129 | 0.146 | 0.160 ✓ | 0.161 ✓ | 0.160 ✓ | 0.177 | 0.160 ✓ | 0.147 | 0.158 |

## L = 1024 · SVM

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.274 | 0.238 | 0.238 | 0.270 | 0.274 ✓ | 0.274 ✓ | 0.274 ✓ | 0.215 | 0.274 ✓ | 0.269 | 0.274 ✓ |
| esc10_4k | 0.264 | 0.227 | 0.218 | 0.257 | 0.265 ✓ | 0.264 ✓ | 0.265 ✓ | 0.217 | 0.264 ✓ | 0.260 | 0.264 ✓ |
| esc10_8k | 0.256 | 0.236 | 0.218 | 0.250 | 0.256 ✓ | 0.256 ✓ | 0.256 ✓ | 0.220 | 0.256 ✓ | 0.247 | 0.255 ✓ |

## L = 1024 · RF

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.340 | 0.245 | 0.246 | 0.291 | 0.340 ✓ | 0.338 | 0.332 | 0.179 | 0.339 ✓ | 0.285 | 0.330 |
| esc10_4k | 0.321 | 0.238 | 0.236 | 0.291 | 0.320 | 0.326 ✓ | 0.317 | 0.186 | 0.322 ✓ | 0.290 | 0.313 |
| esc10_8k | 0.301 | 0.246 | 0.254 | 0.278 | 0.298 | 0.299 ✓ | 0.301 | 0.207 | 0.305 ✓ | 0.279 | 0.296 |

## L = 1024 · LSTM

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.290 | 0.229 | 0.228 | 0.261 | 0.291 ✓ | 0.289 ✓ | 0.290 ✓ | 0.204 | 0.290 ✓ | 0.267 | 0.290 ✓ |
| esc10_4k | 0.260 | 0.230 | 0.233 | 0.247 | 0.259 ✓ | 0.261 ✓ | 0.260 ✓ | 0.216 | 0.260 ✓ | 0.245 | 0.260 ✓ |
| esc10_8k | 0.240 | 0.200 | 0.200 | 0.223 | 0.239 ✓ | 0.240 ✓ | 0.239 ✓ | 0.197 | 0.240 ✓ | 0.226 | 0.240 ✓ |

## L = 1024 · GRU

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.381 | 0.304 | 0.308 | 0.357 | 0.381 ✓ | 0.381 ✓ | 0.379 ✓ | 0.222 | 0.381 ✓ | 0.365 | 0.378 ✓ |
| esc10_4k | 0.378 | 0.293 | 0.300 | 0.350 | 0.377 ✓ | 0.377 ✓ | 0.377 ✓ | 0.262 | 0.378 ✓ | 0.356 | 0.376 ✓ |
| esc10_8k | 0.274 | 0.232 | 0.233 | 0.264 | 0.275 ✓ | 0.274 ✓ | 0.276 ✓ | 0.222 | 0.274 ✓ | 0.267 | 0.277 ✓ |

## L = 1024 · CNN

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP | Sipers · RP sin signo (circular) | Sipers · Signed RP (circular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc10_2k | 0.463 | 0.347 | 0.363 | 0.419 | 0.464 ✓ | 0.463 ✓ | 0.463 ✓ | 0.286 | 0.463 ✓ | 0.422 | 0.460 ✓ |
| esc10_4k | 0.423 | 0.299 | 0.304 | 0.388 | 0.422 ✓ | 0.423 ✓ | 0.423 ✓ | 0.288 | 0.423 ✓ | 0.396 | 0.424 ✓ |
| esc10_8k | 0.482 | 0.340 | 0.373 | 0.425 | 0.480 ✓ | 0.482 ✓ | 0.482 ✓ | 0.355 | 0.482 ✓ | 0.430 | 0.483 |

