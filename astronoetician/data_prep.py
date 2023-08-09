# Creating the dataset for the model and saving it
# %%
import pandas as pd
import torch
#%%
# Reading the data
df = pd.read_csv('../data/minor-planet-names-alphabetical-list.csv', sep=';')
df
# %%
pns = df['Planet name'].str.casefold().value_counts().keys().to_list()
pns
# %%
unqs = sorted(list(set(''.join(pns))))
unqs.insert(0, '|')
len(unqs)
# %%
# creating the mapping
stoi = {c:i for i, c in enumerate(unqs)}
itos = {i:c for i, c in enumerate(unqs)}
print(stoi)
print(itos)
# %%
# creating the dataset
block_size = 3

def create_dataset(pns):
    X, y = [], []
    for pn in pns:
        context = [0] * block_size
        for c in pn + '|':
            X.append(context)
            y.append(stoi[c])
            context = context[1:] + [stoi[c]]
    return torch.tensor(X), torch.tensor(y)

# %%
import random
random.shuffle(pns)
train_pns = pns[:int(0.8*len(pns))]
val_pns = pns[int(0.8*len(pns)):int(0.9*len(pns))]
test_pns = pns[int(0.9*len(pns)):]
# %%
train_X, train_y = create_dataset(train_pns)
val_X, val_y = create_dataset(val_pns)
test_X, test_y = create_dataset(test_pns)

# %%
# saving the dataset
torch.save((train_X, train_y), '../data/train.pt')
torch.save((val_X, val_y), '../data/val.pt')
torch.save((test_X, test_y), '../data/test.pt')

# %%
# saving the mappings
torch.save(stoi, '../data/stoi.pt')
torch.save(itos, '../data/itos.pt')
# %%
