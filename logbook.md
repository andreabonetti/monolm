# monolm: logbook

Latest entries are added on top.

## accuracy

```
$ python accuracy_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 131072 --gpu 43 --threads 8
Examples:                  10
Exact match accuracy:      100.00%
Contains-expected accuracy:100.00%
```

```
$ python accuracy_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 4096 --gpu 33 --threads 8
Examples:                  10
Exact match accuracy:      60.00%
Contains-expected accuracy:100.00%
```

```
$ python accuracy_benchmark.py --model models/mistral-7b-instruct-v0.1.Q5_K_M.gguf --ctx 32768 --gpu 33 --threads 8
Examples:                  10
Exact match accuracy:      40.00%
Contains-expected accuracy:100.00%
```

```
$ python accuracy_benchmark.py --model models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf --ctx 2048 --gpu 22 --threads 8
Examples:                  10
Exact match accuracy:      0.00%
Contains-expected accuracy:100.00%
```


## best timing configurations

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 131072 --gpu 43 --threads 8
TTFT mean: 0.044 s
TTFT std:  0.002 s
TPS mean:  23.94 tok/s
TPS std:   0.19 tok/s
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 4096 --gpu 33 --threads 8
TTFT mean: 0.034 s
TTFT std:  0.001 s
TPS mean:  30.33 tok/s
TPS std:   0.58 tok/s
```

```
$ python timing_benchmark.py --model models/mistral-7b-instruct-v0.1.Q5_K_M.gguf --ctx 32768 --gpu 33 --threads 8
TTFT mean: 0.072 s
TTFT std:  0.001 s
TPS mean:  14.18 tok/s
TPS std:   0.18 tok/s
```

```
$ python timing_benchmark.py --model models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf --ctx 2048 --gpu 22 --threads 8
TTFT mean: 0.014 s
TTFT std:  0.000 s
TPS mean:  70.99 tok/s
TPS std:   4.81 tok/s
```


## experiments: gemma-4-E4B-it-Q4_K_M.gguf

### gpu

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 0 --threads 8
TTFT mean: 0.092s
TTFT std:  0.014s
TPS mean:  13.48
TPS std:   1.11
```

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 2 --threads 8
TTFT mean: 0.088s
TTFT std:  0.012s
TPS mean:  13.38
TPS std:   0.69
```

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 8 --threads 8
TTFT mean: 0.085s
TTFT std:  0.008s
TPS mean:  14.89
TPS std:   0.93
```

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 20 --threads 8
TTFT mean: 0.072 s
TTFT std:  0.007 s
TPS mean:  16.20 tok/s
TPS std:   1.89 tok/s
```

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 50 --threads 8
TTFT mean: 0.043 s
TTFT std:  0.001 s
TPS mean:  24.10 tok/s
TPS std:   0.14 tok/s
```
```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 100 --threads 8
TTFT mean: 0.043 s
TTFT std:  0.003 s
TPS mean:  24.23 tok/s
TPS std:   0.18 tok/s
```


### ctx

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 2048 --gpu 8 --threads 8
TTFT mean: 0.085s
TTFT std:  0.008s
TPS mean:  14.89
TPS std:   0.93
```

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 512 --gpu 8 --threads 8
TTFT mean: 0.092s
TTFT std:  0.014s
TPS mean:  12.64tok/s
TPS std:   2.19tok/s
```

```
$ python timing_benchmark.py --model models/gemma-4-E4B-it-Q4_K_M.gguf --ctx 8192 --gpu 8 --threads 8
TTFT mean: 0.078 s
TTFT std:  0.006 s
TPS mean:  14.34 tok/s
TPS std:   0.89 tok/s
```

## experiments: Phi-3-mini-4k-instruct-q4.gguf

### ctx

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048
TTFT mean: 0.077s
TTFT std:  0.026s
TPS mean:  16.92
TPS std:   1.55
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 512
TTFT mean: 0.079s
TTFT std:  0.017s
TPS mean:  15.63
TPS std:   1.53
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 4096
TTFT mean: 0.078s
TTFT std:  0.011s
TPS mean:  17.45
TPS std:   2.08
```

Changing `ctx` from 512 to 4096 increases the mean TPS of only +11%.

### gpu layers

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048 --gpu 0
TTFT mean: 0.077s
TTFT std:  0.015s
TPS mean:  14.56
TPS std:   1.53
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048 --gpu 1
TTFT mean: 0.076s
TTFT std:  0.005s
TPS mean:  15.89
TPS std:   0.98
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048 --gpu 5
TTFT mean: 0.078s
TTFT std:  0.025s
TPS mean:  16.70
TPS std:   1.17
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048 --gpu 10
TTFT mean: 0.077s
TTFT std:  0.012s
TPS mean:  17.30
TPS std:   0.60
```

### cpu threads

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048 --gpu 0 --threads 4
TTFT mean: 0.047s
TTFT std:  0.004s
TPS mean:  23.42
TPS std:   0.22
```

```
$ python timing_benchmark.py --model models/Phi-3-mini-4k-instruct-q4.gguf --ctx 2048 --gpu 0 --threads 8
TTFT mean: 0.079s
TTFT std:  0.016s
TPS mean:  17.53
TPS std:   1.35
```

## runner.py

If you want low latency:
- keep context small: `n_ctx=1024 or 2048`
- use small models
- push gpu layers: `n_gpu_layers=20` - More layers = faster until memory pressure hits.


## prerequisites

install llama.cpp
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release
```

```
pip install llama-cpp-python
```

⚠️ Important (Mac Metal acceleration)

You must reinstall with Metal enabled so it matches your build:
```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

This is critical. Otherwise you lose GPU acceleration.

