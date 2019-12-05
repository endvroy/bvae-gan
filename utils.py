import torch

''' sample parameters in one `model` '''


def SGHMC(loss, model, Optim=None, steps=int(1e2), lr=1e-2, friction=1e-1, device=None):
    if device is None:
        raise ValueError('device must be specified')
    vList = [torch.randn_like(param, device=device) for param in model.parameters()]
    if Optim is not None:
        optim = Optim(model.parameters())
        for step in range(steps):
            l = loss(model)
            optim.zero_grad()
            l.backward()
            optim.step()
    else:
        for step in range(steps):
            l = loss(model)
            # gradList = torch.autograd.grad(l, model.parameters(), create_graph=True, retain_graph=False)
            gradList = torch.autograd.grad(l, model.parameters())
            for (v, grad, param) in zip(vList, gradList, model.parameters()):
                v.sub_(friction * v + lr * grad + (2 * lr * friction) * torch.randn_like(v))
                param.data.add_(v)


''' sample parameters in the whole `modelBatchUnion` '''


def Gibbs(lossList, modelBatchUnion, dataBatchList, epochs, *argsMC, **kwargsMC):
    assert (len(lossList) == len(modelBatchUnion))
    for epoch in range(epochs):
        print('Epoch:', epoch + 1)
        for (batch, dataBatch) in enumerate(iter(dataBatchList)):
            for (k, (_loss, modelBatch)) in enumerate(zip(lossList, modelBatchUnion)):
                modelBatchOthers = [modelBatch for (j, modelBatch) in enumerate(modelBatchUnion) if j != k]
                loss = lambda model: _loss(model, dataBatch, *modelBatchOthers)
                for (m, model) in enumerate(modelBatch):
                    SGHMC(loss, model, *argsMC, **kwargsMC)
                    print('.', end='', flush=True)
                print('-', end='', flush=True)
            print('=', end='', flush=True)
        print()
