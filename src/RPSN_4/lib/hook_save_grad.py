


# grads = {}
def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook