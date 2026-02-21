import time

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

R_S = 1.0  # Ensure this is a float

HEIGHT = 720
RESOLUTION = (HEIGHT * 16 // 9, HEIGHT)

camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())  # mutable scalar vec3
look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
fov = ti.field(dtype=ti.f32, shape=())

output_image = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)


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


@ti.func
def black_hole_ode(u, v):
    du_dphi = v
    dv_dphi = 1.5 * R_S * u ** 2 - u
    return du_dphi, dv_dphi


@ti.func
def integrate_rk4(u, v, dphi):
    k1_u, k1_v = black_hole_ode(u, v)
    k2_u, k2_v = black_hole_ode(u + 0.5 * dphi * k1_u, v + 0.5 * dphi * k1_v)
    k3_u, k3_v = black_hole_ode(u + 0.5 * dphi * k2_u, v + 0.5 * dphi * k2_v)
    k4_u, k4_v = black_hole_ode(u + dphi * k3_u, v + dphi * k3_v)
    
    u_next = u + (dphi / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    v_next = v + (dphi / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    
    return u_next, v_next


IntegrationResult = ti.types.struct(
    u=ti.f32,
    v=ti.f32,
    phi=ti.f32,
    hit_photon_sphere=ti.i32,
    hit_range_limit=ti.i32
)


@ti.func
def perform_integration(u_0, v_0, dphi, steps) -> IntegrationResult:
    u, v = u_0, v_0
    phi = 0.0
    
    inv_photon_sphere = 1.0 / (1.5 * R_S)
    inv_range_limit = 1.0 / (50.0 * R_S)
    
    hit_photon_sphere = 0
    hit_range_limit = 0
    
    for _ in range(steps):
        u, v = integrate_rk4(u, v, dphi)
        phi += dphi
        
        if u > inv_photon_sphere:
            hit_photon_sphere = 1
            break
        if u < inv_range_limit:
            hit_range_limit = 1
            break
    
    return IntegrationResult(u, v, phi, hit_photon_sphere, hit_range_limit)


@ti.kernel
def render():
    for x, y in ti.ndrange(RESOLUTION[0], RESOLUTION[1]):
        ray_origin = camera_pos[None]
        ray_dir = start_ray(x, y)
        
        normal = tm.normalize(tm.cross(ray_origin, ray_dir))
        
        e_r = tm.normalize(ray_origin)
        e_t = tm.normalize(tm.cross(normal, e_r))
        
        u_0 = 1.0 / tm.length(ray_origin)
        v_0 = -tm.dot(ray_dir, e_r) / (tm.length(ray_origin) * tm.dot(ray_dir, e_t))
        
        # Keyword arguments are not supported in Taichi device functions
        result = perform_integration(u_0, v_0, 0.001, 10000)
        
        u_final = result.u
        v_final = result.v
        phi_final = result.phi
        
        final_pos_3d = (1.0 / u_final) * (e_r * tm.cos(phi_final) + e_t * tm.sin(phi_final))
        
        v_term = (-v_final / u_final ** 2) * (e_r * tm.cos(phi_final) + e_t * tm.sin(phi_final))
        u_term = (1.0 / u_final) * (-e_r * tm.sin(phi_final) + e_t * tm.cos(phi_final))
        final_dir_3d = tm.normalize(v_term + u_term)
        
        output_image[x, y] = final_dir_3d * 0.5 + 0.5 if result.hit_photon_sphere == 0 else tm.vec3(0.0, 0.0, 0.0)


@ti.kernel
def init():
    camera_pos[None] = tm.vec3(0.0, 0.0, -5.0)
    look_at[None] = tm.vec3(0.0, 0.0, 0.0)
    fov[None] = tm.radians(90.0)


def main():
    init()
    
    start_time = time.perf_counter()
    render()
    end_time = time.perf_counter()
    print(f"Render time: {end_time - start_time:.2f} seconds")
    
    output_numpy = output_image.to_numpy()
    
    # Show output image
    gui = ti.GUI("Black Hole Ray Tracing", RESOLUTION)
    while gui.running:
        gui.set_image(output_numpy)
        gui.show()


if __name__ == "__main__":
    main()