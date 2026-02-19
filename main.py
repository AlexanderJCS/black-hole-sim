import taichi as ti
import taichi.math as tm
import numpy as np
import imageio.v3 as iio

ti.init(arch=ti.gpu)

RESOLUTION = (800, 600)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)


camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())  # mutable scalar vec3
look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
fov = ti.field(dtype=ti.f32, shape=())

@ti.func
def get_camera_basis():
    w = (camera_pos[None] - look_at[None]).normalized()
    up = tm.vec3(0.0, 1.0, 0.0)
    if abs(w.dot(up)) > 0.999:
        up = tm.vec3(0.0, 0.0, 1.0)
    u = up.cross(w).normalized()
    v = w.cross(u)
    return u, v, w


@ti.func
def start_ray(x: int, y: int) -> tm.vec3:
    u, v, w = get_camera_basis()
    aspect = RESOLUTION[0] / RESOLUTION[1]
    half_height = ti.tan(fov[None] / 2.0)
    half_width = half_height * aspect

    pixel_u = (x + 0.5) / RESOLUTION[0]
    pixel_v = (y + 0.5) / RESOLUTION[1]

    horizontal = 2 * half_width * u
    vertical = 2 * half_height * v
    lower_left = camera_pos[None] - half_width * u - half_height * v - w

    return (lower_left + pixel_u * horizontal + pixel_v * vertical - camera_pos[None]).normalized()


@ti.kernel
def init():
    for i, j in pixels:
        pixels[i, j] = 0.0


@ti.func
def dd_dt(ray_pos: tm.vec3, ray_dir: tm.vec3, r: tm.vec3, mass: float) -> tm.vec3:
    # Use the equation:
    # d_new = d - (r * dt * M / |r|^3)
    # Therefore:
    # dd/dt = -Mr/|r|^3
    return -mass * r / (r.norm()**3 + 1e-6)  # Add small epsilon to avoid division by zero


@ti.func
def integrate_rk4(ray_pos: tm.vec3, ray_dir: tm.vec3, dt: float, r: tm.vec3, mass: float) -> (tm.vec3, tm.vec3):
    k1 = dd_dt(ray_pos, ray_dir, r, mass)
    k2 = dd_dt(ray_pos + ray_dir * (dt / 2), ray_dir + k1 * (dt / 2), r, mass)
    k3 = dd_dt(ray_pos + ray_dir * (dt / 2), ray_dir + k2 * (dt / 2), r, mass)
    k4 = dd_dt(ray_pos + ray_dir * dt, ray_dir + k3 * dt, r, mass)
    
    new_dir = ray_dir + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    new_pos = ray_pos + new_dir * dt
    
    return new_pos, new_dir.normalized()


@ti.func
def get_ray_color(ray_dir: tm.vec3) -> tm.vec3:
    # Gets the ray color from the direction. Make a checkerboard pattern using values (0.5, 0.5, 0.5,) and (1, 1, 1).
    u = 0.5 * (tm.atan2(ray_dir.z, ray_dir.x) / tm.pi + 1.0)
    v = 0.5 * (tm.asin(ray_dir.y) / (tm
.pi / 2) + 1.0)
    checker = (int(u * 10) % 2) ^ (int(v * 10) % 2)
    return tm.vec3(0.5, 0.5, 0.5) if checker == 0 else tm.vec3(1.0, 1.0, 1.0)


@ti.func
def march(ray_origin: tm.vec3, ray_dir: tm.vec3, distance: float, dt: float) -> tm.vec3:
    current_pos = ray_origin
    current_dir = ray_dir

    bh = tm.vec3(0.0, 0.0, 0.0)
    mass = 2.5
    
    swartzschild_radius = mass
    absorbed = False

    t = 0.0
    while t < distance:
        dist_to_bh = (current_pos - bh).norm()
        if dist_to_bh < swartzschild_radius:
            absorbed = True
            break
        
        current_pos, current_dir = integrate_rk4(current_pos, current_dir, dt, bh - current_pos, mass)
        t += dt
        
    if absorbed:
        current_dir = tm.vec3(0.0, 0.0, 0.0)  # Absorbed rays have no direction

    return get_ray_color(current_dir)


@ti.kernel
def render():
    for i, j in pixels:
        ray_origin = camera_pos[None]
        ray_dir = start_ray(i, j)
        
        pixels[i, j] = march(ray_origin, ray_dir, 50, 0.1) * 0.5 + 0.5
        # pixels[i, j] = ray_dir * 0.5 + 0.5


def main():
    gui = ti.GUI("Black Hole", res=RESOLUTION)
    
    camera_pos[None] = tm.vec3(0.0, 0.0, 20.0)
    look_at[None] = tm.vec3(0.0, 0.0, 0.0)
    fov[None] = 60.0 * ti.math.pi / 180.0
    
    while gui.running:
        render()
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
