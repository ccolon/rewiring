"""Random parameter generation for the rewiring model.

Public API:
    EPSILON                   -- numerical tolerance used across the package
    draw_random_vector_normal -- bounded normal draws
    generate_parameter        -- vector of (homogeneous|uniform|normal) draws
    generate_a_parameter      -- labor share `a` with the constraint a*b < 1
"""

import numpy as np


EPSILON = 1e-10


def draw_random_vector_normal(mean, sigma, n, bound_min=None, bound_max=None):
    """Draw n values from a normal distribution, clipped to optional bounds."""
    values = np.random.normal(mean, sigma, n)
    if bound_min is not None:
        values = np.maximum(values, bound_min)
    if bound_max is not None:
        values = np.minimum(values, bound_max)
    return values


def generate_parameter(config, n, param_name='param', verbose=True):
    """Generate a parameter vector based on a mode-config dict.

    config keys:
        mode : 'homogeneous' | 'uniform' | 'normal'
        homogeneous : value
        uniform     : min, max
        normal      : mean, sigma, [bound_min, bound_max]
    """
    mode = config.get('mode', 'homogeneous')

    if mode == 'homogeneous':
        value = config.get('value', 1.0)
        values = np.full(n, value)
        if verbose:
            print(f"  {param_name} (homogeneous): value={value}")

    elif mode == 'uniform':
        min_val = config.get('min', 0.0)
        max_val = config.get('max', 1.0)
        values = np.random.uniform(min_val, max_val, n)
        if verbose:
            print(f"  {param_name} (uniform): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    elif mode == 'normal':
        mean = config.get('mean', 1.0)
        sigma = config.get('sigma', 0.1)
        bound_min = config.get('bound_min', None)
        bound_max = config.get('bound_max', None)
        values = draw_random_vector_normal(mean, sigma, n, bound_min, bound_max)
        if verbose:
            print(f"  {param_name} (normal): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    else:
        raise ValueError(f"Unknown mode '{mode}' for parameter {param_name}")

    return values


def generate_a_parameter(a_config, b, n, verbose=True):
    """Generate labor share `a` with the constraint a*b < 1.

    For each firm i, max_a_i = min((1 - eps)/b[i], 1 - eps).
    """
    mode = a_config.get('mode', 'homogeneous')
    eps = 0.05

    if mode == 'homogeneous':
        value = a_config.get('value', 0.5)
        values = np.minimum(value, (1 - eps) / b)
        if verbose:
            print(f"  a (homogeneous): value={value}, adjusted range=[{values.min():.4f}, {values.max():.4f}]")

    elif mode == 'uniform':
        min_val = a_config.get('min', 0.1)
        max_vals = np.minimum(a_config.get('max', 0.9), (1 - eps) / b)
        values = np.array([np.random.uniform(min_val, max_val) for max_val in max_vals])
        if verbose:
            print(f"  a (uniform): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    elif mode == 'normal':
        mean = a_config.get('mean', 0.5)
        sigma = a_config.get('sigma', 0.1)
        min_a = eps
        max_vals = np.minimum((1 - eps) / b, 1 - eps)
        values = np.array([
            draw_random_vector_normal(mean, sigma, 1, min_a, max_val)[0]
            for max_val in max_vals
        ])
        if verbose:
            print(f"  a (normal): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    else:
        raise ValueError(f"Unknown mode '{mode}' for parameter a")

    return values
