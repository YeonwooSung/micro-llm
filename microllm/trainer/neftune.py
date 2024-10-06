import torch


#   <https://github.com/huggingface/trl/issues/870>
def NEFTune(model, noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha/torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)
        return new_func

    ##### NOTE: this is for a LLaMA model #####
    ##### For a different model, you need to change the attribute path to the embedding #####
    model.model.tok_embeddings.forward = noised_embed(model.model.tok_embeddings.forward, noise_alpha)
    return model
