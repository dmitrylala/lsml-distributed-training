# LSML: Distributed training

Код для семинаров по распределенному обучению LLM в рамках курса Large Scale ML.

Сетап:
```bash
source .env && make vendor
```

В качестве примера будем учить GPT2 (124M параметров).

Запуск обучения на одной карте:
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_single.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2 \
    -e test01-single
```

На 4 картах в DDP:
```bash
export TORCHELASTIC_ERROR_FILE=error.json; \
export OMP_NUM_THREADS=1; \
export CUDA_VISIBLE_DEVICES=0,1,2,3; \
uv run torchrun \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir logs \
    scripts/train_ddp.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2 \
    -e test01-ddp
```
