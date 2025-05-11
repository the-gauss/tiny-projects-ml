import json
import torch

class BaseLine(torch.nn.Module):
    def __init__(self, n_layers = 2 ,activation = torch.nn.ReLU()):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        print(self.flatten)
        self.activation = activation
        self.n_layers = n_layers
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3*224*224, 512),
            self.activation,
            *[torch.nn.Linear(512, 512), self.activation]*self.n_layers,
            torch.nn.Linear(512,102)
        )
        print(self.linear_relu_stack)

    def forward(self, x):
        x = self.flatten(x) # Flattens the second dimension (dim=1), leaves dim=0 intact
        logits = self.linear_relu_stack(x)  # Returns unnormalized [-infty, infty] outputs of the final layer
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    device = torch.accelerator.current_accelerator().type
    size = len(dataloader)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
       X, y = X.to(device), y.to(device)

       # Forward prop
       y_pred = model(X)
       loss = loss_fn(y_pred, y)
        # Backprop
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

       loss = loss.item()
       if batch >= len(dataloader):
           print(f'Batch: {batch} \t Loss: {loss}\n')

def test_loop(dataloader, model, loss_fn):
    device = torch.accelerator.current_accelerator().type
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():   # during testing, we don't wanna waste time and memory computing grads
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item() # loss is tensor, need to extract the item for +=
            accuracy += (y_pred.argmax(1)==y).type(torch.float).sum().item()
        test_loss /= size
        accuracy /= size
        print(f'Test Accuracy: {accuracy}\n')