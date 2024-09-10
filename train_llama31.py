"""
This code is from "https://github.com/karpathy/nano-llama31".
Credits to @karpathy.


micro-Llama 3.1
Simpler version you can just forward on 1 GPU, without torchrun.
Changes:
- replace ColumnParallelLinear -> Linear
- replace RowParallelLinear -> Linear
- replace VocabParallelEmbedding -> Embedding

Run example:

python train_llama31.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
"""

import os
import fire
import time
from pathlib import Path
from typing import List
import torch
import sys

sys.path.append(".")

# custom modules
from microllm.llama.llama31.model import Llama31 as Llama
from microllm.llama.llama31.dataloader import DistributedShardedDataLoader


def main(
    ckpt_dir: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B",
    tokenizer_path: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model",
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_gen_len: int = 256,
    max_batch_size: int = 8,
    flash: bool = True,
):
    # load the val data shard
    data_loader = DistributedShardedDataLoader(
        filename_pattern="tinystories/*_val.bin",
        B=max_batch_size,
        T=max_seq_len,
        process_rank=0,
        num_processes=1,
    )

    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        flash=flash,
    )

    total_batch_size = max_batch_size * max_seq_len
    print(f"total_batch_size: {total_batch_size}")

    # super simple training loop to start
    model = llama.model
    model.train()
    optimizer = model.configure_optimizers(learning_rate=1e-5, weight_decay=0.0)
    for step in range(20):
        optimizer.zero_grad()
        x, y = data_loader.next_batch()
        x, y = x.cuda(), y.cuda()
        loss = model.forward_loss(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {step}, loss: {loss.item()}")

    # and now generate
    model.eval()
    prompts: List[str] = [
        "Once upon a time",
        "One day",
        "Lily and George were best friends",
        "On a dark and stormy night",
    ]

    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)
    t0 = time.time()
    results = llama.text_completion(
        prompts,
        sample_rng=sample_rng,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    t1 = time.time()
    print(f"Generated in {t1 - t0:.2f} seconds")
    for prompt, result in zip(prompts, results):
        print(prompt, end="") # AK: change end="\n" to end=""
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
