# generate planet names from the model
# %%
import torch
import torch.nn.functional as F
# %%
# loading the model
parameters = torch.load('../parameters/astronoetician.pt')
C, W1, B1, W2, B2 = parameters

# %%
# loading the mappings
stoi = torch.load('../data/stoi.pt')
itos = torch.load('../data/itos.pt')

# %%
# generating planet names
def generate_planet_name(model, stoi, itos, block_size=3, max_len = 20):
    for _ in range(max_len):
        out = []
        context = [0] * block_size

        while True:
            emb = C[torch.tensor(context)]
            h = torch.tanh(emb.view(1, -1) @ W1 + B1)
            logits = h @ W2 + B2
            p = F.softmax(logits, dim=-1)
            c = torch.multinomial(p, 1).item()
            out.append(itos[c])
            context = context[1:] + [c]
            if c == 0:
                break
    
        print(''.join(out[:-1]).capitalize())
# %%
generate_planet_name(parameters, stoi, itos)
# %%
