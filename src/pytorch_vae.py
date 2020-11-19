"""alias to PyTorch-VAE to normalize the module name"""
x = __import__("PyTorch-VAE")
globals().update(vars(x))