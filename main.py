import time

import numpy as np

import taichi as ti
import taichi.math as tm

from PIL import Image

ti.init(arch=ti.gpu)

R_S = 1.0  # Ensure this is a float
R_MS = 3 * R_S  # https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit

accretion_absorption = 0.9  # absorption coefficient
accretion_emission = 0.5  # emiited radiance per unit length

HEIGHT = 720
RESOLUTION = (HEIGHT * 16 // 9, HEIGHT)

camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())  # mutable scalar vec3
look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
fov = ti.field(dtype=ti.f32, shape=())

output_image = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)

pil = Image.open("resources/refined_galactic_plane_and_nebulae_1.png").convert("RGBA")
np_img = np.asarray(pil, dtype=np.float32) / 255.0   # shape (H, W, 4), 0..1 floats
np_img = np.transpose(np_img, (1, 0, 2))            # shape (W, H, 4) to match your output_image[x, y]

W, H = np_img.shape[0], np_img.shape[1]

# create a Taichi Vector field and copy the numpy data into it
spheremap_field = ti.Vector.field(4, dtype=ti.f32, shape=(W, H))
spheremap_field.from_numpy(np_img)   # copies data to Taichi device memory


@ti.func
def sample_spheremap_uv(u: ti.f32, v: ti.f32) -> tm.vec3:
    # wrap u, clamp v
    u = u - ti.floor(u)                      # wrap to [0,1)
    v = ti.min(ti.max(v, 0.0), 1.0)          # clamp

    # continuous coordinates
    x = u * (W - 1)
    y = v * (H - 1)

    x0 = ti.cast(ti.floor(x), ti.i32)
    y0 = ti.cast(ti.floor(y), ti.i32)
    x1 = (x0 + 1) % W
    y1 = ti.min(y0 + 1, H - 1)

    sx = x - x0
    sy = y - y0

    c00 = spheremap_field[x0, y0]
    c10 = spheremap_field[x1, y0]
    c01 = spheremap_field[x0, y1]
    c11 = spheremap_field[x1, y1]

    c0 = c00 * (1.0 - sx) + c10 * sx
    c1 = c01 * (1.0 - sx) + c11 * sx
    c = c0 * (1.0 - sy) + c1 * sy

    return tm.vec3(c.x, c.y, c.z)


@ti.func
def sample_spheremap(ray_dir: ti.types.vector(3, dtype=ti.f32)) -> tm.vec3:
    theta = ti.acos(ray_dir.y)
    phi = ti.atan2(ray_dir.z, ray_dir.x)
    u = (phi + tm.pi) / (2 * tm.pi)
    v = theta / tm.pi
    return sample_spheremap_uv(u, v)


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
def accretion_emissivity(pos: ti.types.vector(3, dtype=ti.f32)):
    r = pos.norm()
    base = ti.max(10.0 * tm.log(r) / (r * r), 0.0)

    height = 0.25
    sigma = height / 4.0
    vertical = tm.exp(-0.5 * (pos.y / sigma) ** 2)

    return base * vertical


@ti.func
def accretion_density(pos: ti.types.vector(3, dtype=ti.f32)):
    # accretion_absorption if inside the cylinder (with a hole in the middle), otherwise 0
    # the cylinder's inner radius is R_MS, outer radius is R_S * 10, and height is 0.25
    
    height = 0.25
    
    density = 0.0
    r = tm.sqrt(pos.x ** 2 + pos.z ** 2)
    if R_MS < r < R_S * 10 and abs(pos.y) < height:
        density = accretion_absorption
        
        # Multiply the density by the normal distribution on the y-axis. Sigma = height / 4.
        sigma = height / 4.0
        density *= tm.exp(-0.5 * (abs(pos.y) / sigma) ** 2)
    
    return density


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
    light=ti.f32,
    hit_photon_sphere=ti.i32,
    hit_range_limit=ti.i32
)


@ti.func
def perform_integration(u_0, v_0, max_dphi, max_steps, e_r, e_t) -> IntegrationResult:
    u, v = u_0, v_0
    phi = 0.0
    inv_photon_sphere = 1.0 / (1.5 * R_S)
    inv_range_limit = 1.0 / (50.0 * R_S)
    hit_photon_sphere = 0
    hit_range_limit = 0
    transmittance = 1.0
    light = 0.0

    # initial position
    prev_coords_3d = (1.0 / u) * (e_r * tm.cos(phi) + e_t * tm.sin(phi))
    prev_emiss = accretion_emissivity(prev_coords_3d)

    ds_target = 0.02  # spatial step target; reduce for higher quality
    for i in range(max_steps):
        # choose angle step so arc length ~ ds_target: dphi = ds_target / r = ds_target * u
        dphi_local = ds_target * u
        
        # clamp for stability
        dphi_local = tm.clamp(dphi_local, 1e-5, max_dphi)

        # integrate by adaptive angle
        u_next, v_next = integrate_rk4(u, v, dphi_local)

        phi_next = phi + dphi_local
        coords_3d = (1.0 / u_next) * (e_r * tm.cos(phi_next) + e_t * tm.sin(phi_next))
        ds = tm.length(coords_3d - prev_coords_3d)

        # sample density with smoothed edges (see next fix)
        rho = accretion_density(coords_3d)

        if rho > 0.0:
            # trap rule for emission: avg(prev_emiss, curr_emiss) * transmittance * ds
            curr_emiss = accretion_emissivity(coords_3d)
            light += 0.5 * (prev_emiss + curr_emiss) * transmittance * ds
            prev_emiss = curr_emiss

            transmittance *= tm.exp(-rho * ds)

        prev_coords_3d = coords_3d
        u, v, phi = u_next, v_next, phi_next

        if transmittance < 1e-3:
            break
        if u > inv_photon_sphere:
            hit_photon_sphere = 1
            break
        if u < inv_range_limit:
            hit_range_limit = 1
            break

    return IntegrationResult(u, v, phi, light, hit_photon_sphere, hit_range_limit)


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
        result = perform_integration(u_0, v_0, 0.01, 5000, e_r, e_t)
        
        u_final = result.u
        v_final = result.v
        phi_final = result.phi
        
        cos_phi_final = tm.cos(phi_final)
        sin_phi_final = tm.sin(phi_final)
        
        final_pos_3d = (1.0 / u_final) * (e_r * cos_phi_final + e_t * sin_phi_final)
        
        v_term = (-v_final / u_final ** 2) * (e_r * cos_phi_final + e_t * sin_phi_final)
        u_term = (1.0 / u_final) * (-e_r * sin_phi_final + e_t * cos_phi_final)
        final_dir_3d = tm.normalize(v_term + u_term)
        
        output_image[x, y] = result.light


@ti.kernel
def init():
    camera_pos[None] = tm.vec3(15.0, 0.0, 0.0)
    look_at[None] = tm.vec3(0.0, 0.0, 0.0)
    fov[None] = tm.radians(90.0)


def main():
    init()
    render()
    
    start_time = time.perf_counter()
    render()
    end_time = time.perf_counter()
    print(f"Render time: {(end_time - start_time) * 1000:.7f} ms")
    
    output_numpy = output_image.to_numpy()
    
    # Show output image
    gui = ti.GUI("Black Hole Ray Tracing", RESOLUTION)
    while gui.running:
        gui.set_image(output_numpy)
        gui.show()


if __name__ == "__main__":
    main()
