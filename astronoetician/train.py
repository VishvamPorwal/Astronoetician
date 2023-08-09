# Training script for the Astronoetician model
# %%
import torch
import torch.nn.functional as F
# %%
# Loading the dataset
train_X, train_y = torch.load('../data/train.pt')
val_X, val_y = torch.load('../data/val.pt')
test_X, test_y = torch.load('../data/test.pt')
# %%
# Loading the mappings
stoi = torch.load('../data/stoi.pt')
itos = torch.load('../data/itos.pt')
# %%
# Parameters
C = torch.randn((len(stoi), 2)) # the embedding matrix
# hidden layer 1
W1 = torch.randn((6, 200)) # the first weight matrix
B1 = torch.randn(200) # the first bias matrix
# Output layer
W2 = torch.randn(200, len(stoi)) # the second weight matrix
B2 = torch.randn(len(stoi)) # the second bias matrix
parameters = [C, W1, B1, W2, B2]
sum(p.nelement() for p in parameters)
# %%
# set requires_grad=True for all parameters
for p in parameters:
    p.requires_grad_()
# %%
step = []
losses = []
# %%
# training loop
for epoch in range(10001):
    # construct minibatch
    i = torch.randint(0, train_X.shape[0], (32,))

    # forward pass
    emb = C[train_X[i]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + B1)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, train_y[i])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    for p in parameters:
        p.data -= 0.01 * p.grad
    
    # logging
    step.append(epoch)
    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f'Epoch {epoch} | Loss {loss.item()}')
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.xlabel('Epoch') 
plt.ylabel('Log Loss')
plt.title('Training Log Loss vs Epoch')
plt.plot(torch.tensor(step), torch.log(torch.tensor(losses)), label='log loss', color='red', linewidth=2, alpha=0.5)
plt.legend()
plt.savefig('../images/loss_vs_epoch.png')
# %%
# Saving the model
torch.save(parameters, '../parameters/astronoetician.pt')
# %%
# Validation Loss
emb = C[val_X]
h = torch.tanh(emb.view(-1, 6) @ W1 + B1)
logits = h @ W2 + B2
loss = F.cross_entropy(logits, val_y)
print(f'Validation Loss: {loss.item()}')

# %%
# Test Loss
emb = C[test_X]
h = torch.tanh(emb.view(-1, 6) @ W1 + B1)
logits = h @ W2 + B2
loss = F.cross_entropy(logits, test_y)
print(f'Test Loss: {loss.item()}')