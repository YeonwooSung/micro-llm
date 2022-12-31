
CONFIG_DICT = {
    # names follow the huggingface naming conventions
    # GPT-1
    'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    # GPT-2 configs
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    # Gophers
    'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
}
