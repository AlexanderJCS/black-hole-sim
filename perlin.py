import numpy as np
import taichi as ti
import taichi.math as tm


possible_vectors_cpu = np.array([
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
    [np.sqrt(2) / 2, np.sqrt(2) / 2],
    [-np.sqrt(2) / 2, np.sqrt(2) / 2],
    [-np.sqrt(2) / 2, -np.sqrt(2) / 2],
    [np.sqrt(2) / 2, -np.sqrt(2) / 2]
], dtype=np.float32)


possible_vectors_gpu = ti.Vector.field(2, dtype=ti.f32, shape=possible_vectors_cpu.shape[0])
possible_vectors_gpu.from_numpy(possible_vectors_cpu)

POSSIBLE_VECTORS_LEN = len(possible_vectors_cpu)


@ti.func
def pcg_hash(x: ti.types.uint32) -> ti.types.uint32:
    """
    The PCG hash, as recommended by Nathan Reed in "Hash Functions for GPU Rendering" (2021)
    https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/

    Used for Perlin noise generation.

    :param x: Unsigned integer input to the hash function
    :return: The hashed value.
    """
    
    state = x * ti.uint32(747796405) + ti.uint32(2891336453)
    word = ((state >> ((state >> ti.uint32(28)) + ti.uint32(4))) ^ state) * ti.uint32(277803737)
    return (word >> ti.uint32(22)) ^ word


@ti.func
def get_gradient_vector(x: int, y: int) -> ti.types.vector(2, ti.f32):
    """
    Get the gradient vector for the grid point at (x, y) using the PCG hash function.

    :param x: The x-coordinate of the grid point.
    :param y: The y-coordinate of the grid point.
    :return: The gradient vector for the grid point.
    """
    # First combine the coordinates into a single integer. For a 2D plane made of positive integers, we can use the Cantor pairing function:
    #  pi(x, y) = (x + y)(x + y + 1)/2 + y
    # Since x or y can be negative, we first need to map negative integers into nonnegative integers using a bijection:
    # f(n) = 2n if n >= 0 else -2n - 1
    
    x_mapped = 2 * x if x >= 0 else -2 * x - 1
    y_mapped = 2 * y if y >= 0 else -2 * y - 1
    
    paired = (x_mapped + y_mapped) * (x_mapped + y_mapped + 1) // 2 + y_mapped
    hashed = pcg_hash(ti.uint32(paired))
    
    idx = ti.cast(hashed % ti.uint32(POSSIBLE_VECTORS_LEN), ti.i32)
    return possible_vectors_gpu[idx]


@ti.func
def fade(t: ti.f32) -> ti.f32:
    """
    The fade function used for interpolation in Perlin noise.

    :param t: The input value to the fade function, typically in the range [0, 1].
    :return: The output of the fade function.
    """
    return 6.0 * t ** 5.0 - 15.0 * t ** 4.0 + 10.0 * t ** 3.0


@ti.func
def perlin_noise(x: ti.f32, y: ti.f32) -> ti.f32:
    """
    Calculate the Perlin noise value at the point (x, y).
    :param x: The x-coordinate
    :param y: The y-coordinate
    :return: Perlin noise value at (x, y)
    """
    
    x0 = int(tm.floor(x))
    x1 = x0 + 1
    y0 = int(tm.floor(y))
    y1 = y0 + 1
    
    topleft_gradient = get_gradient_vector(x0, y0)
    topright_gradient = get_gradient_vector(x1, y0)
    bottomleft_gradient = get_gradient_vector(x0, y1)
    bottomright_gradient = get_gradient_vector(x1, y1)
    
    topleft_distance = tm.vec2(x - x0, y - y0)
    topright_distance = tm.vec2(x - x1, y - y0)
    bottomleft_distance = tm.vec2(x - x0, y - y1)
    bottomright_distance = tm.vec2(x - x1, y - y1)
    
    topleft_dot = tm.dot(topleft_gradient, topleft_distance)
    topright_dot = tm.dot(topright_gradient, topright_distance)
    bottomleft_dot = tm.dot(bottomleft_gradient, bottomleft_distance)
    bottomright_dot = tm.dot(bottomright_gradient, bottomright_distance)
    
    u = fade(x - x0)
    v = fade(y - y0)
    
    lerp_top = (1 - u) * topleft_dot + u * topright_dot
    lerp_bottom = (1 - u) * bottomleft_dot + u * bottomright_dot
    
    return (1 - v) * lerp_top + v * lerp_bottom
