from .hsvmamba import hsvmamba_tiny_s1l8


def build_model(*args, **kwargs):
    return hsvmamba_tiny_s1l8()
