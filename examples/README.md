# Simple readme on running LMCache v0.2.0 w. vllm
## List of repos that require `pip install -e .`
```
VLLM: https://github.com/KuntaiDu/vllm/tree/jiayi-dev-v2/
LMCache-VLLM: https://github.com/LMCache/lmcache-vllm/tree/dev-v2
LMCache-server: https://github.com/LMCache/lmcache-server/tree/dev-v2
LMCache: https://github.com/LMCache/LMCache/tree/dev-v2
```
Note that LMCache-server and LMCache are unchanged. I created these two branches in case we make modifications to these two repos in v0.1.0 that might affect v0.2.0.

## To run the example:
### Step 1: Start VLLM and LMCache instances (processes)
```
bash example.sh
```
### Step 2: Send two consecutiv requests to VLLM with the same prefix
```
python test_long_prefix.py
```