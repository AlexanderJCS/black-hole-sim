import numpy as np

import taichi as ti
import taichi.math as tm

from PIL import Image

ti.init(arch=ti.gpu)

real_G = 6.6743e-11
real_C = 299792458.0
real_M = 20 * 1.989e30  # 20 solar masses
real_GM = real_G * real_M

R_S = 1.0

# Now suppose that R_S is some manageable number, therefore 2GM/c^2 = 1. Scale GM and c accordingly:
real_R_s = 2 * real_GM / real_C ** 2
scale = R_S / real_R_s

scaled_GM = real_GM * scale
scaled_C = real_C * scale ** 0.5

R_MS = 3 * R_S  # https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit

accretion_absorption = 1.0  # absorption coefficient
brightness_multiplier = 50.0

HEIGHT = 720
RESOLUTION = (HEIGHT * 16 // 9, HEIGHT)

camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())  # mutable scalar vec3
look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
fov = ti.field(dtype=ti.f32, shape=())

linear_image = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
srgb_image = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)

pil = Image.open("resources/refined_galactic_plane_and_nebulae_1.png").convert("RGBA")
np_img = np.asarray(pil, dtype=np.float32) / 255.0   # shape (H, W, 4), 0..1 floats
np_img = np.transpose(np_img, (1, 0, 2))            # shape (W, H, 4) to match your output_image[x, y]

W, H = np_img.shape[0], np_img.shape[1]

# create a Taichi Vector field and copy the numpy data into it
spheremap_field = ti.Vector.field(4, dtype=ti.f32, shape=(W, H))
spheremap_field.from_numpy(np_img)   # copies data to Taichi device memory

color_csv = np.loadtxt("resources/blackbody_color.csv", delimiter=",", skiprows=1).astype(np.float32)
# The temperature as a function of row index is a linear function following the formula:
#   temp = 1000 + index * 100
color_srgb = color_csv[:, 1:4]  # sRGB
color_linear = np.where(color_srgb <= 0.04045, color_srgb / 12.92, ((color_srgb + 0.055) / 1.055) ** 2.4)

color_linear_gpu = ti.Vector.field(3, dtype=ti.f32, shape=color_linear.shape[0])
color_linear_gpu.from_numpy(color_linear)


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
def accretion_density(pos: ti.types.vector(3, dtype=ti.f32), height):
    # accretion_absorption if inside the cylinder (with a hole in the middle), otherwise 0
    # the cylinder's inner radius is R_MS, outer radius is R_S * 10, and height is 0.25
    
    density = 0.0
    r = tm.sqrt(pos.x ** 2 + pos.z ** 2)
    if R_MS < r and abs(pos.y) < height:
        density = tm.pow(r, -0.75) * tm.pow(1.0 - tm.sqrt(R_MS / r), 0.25)
        density *= accretion_absorption
    
    return density


@ti.func
def orbital_velocity(r):
    # Relativistic orbital velocity formula for a Schwarzschild black hole
    # v = sqrt(G * M / r) / sqrt(1 - 2 * G * M / (r * c^2))
    # But since r is in units of R_S = 2GM/c^2, we can simplify.
    # Let r = x * R_S, and since R_S = 2GM/c^2
    # Then we get v = c (sqrt(1 / 2(r - 1)))

    x = r / R_S

    return scaled_C * tm.sqrt(1 / (2 * (x - 1))) if x > 1 else 0.0  # no stable orbits inside R_S, so return 0


@ti.func
def orbital_velocity_direction(pos: ti.types.vector(3, dtype=ti.f32)):
    e_r = tm.normalize(pos)
    up = tm.vec3(0.0, 1.0, 0.0)
    e_t = tm.normalize(tm.cross(up, e_r))  # azimuthal (prograde) direction
    return e_t


@ti.func
def disk_temperature(r, temp_scale=32000.0):
    r_in = R_MS
    
    temp = 0.0
    if r > r_in:
        temp = tm.pow(r, -0.75) * tm.pow(1.0 - tm.sqrt(r_in / r), 0.25)
    
    return temp * temp_scale


@ti.func
def temp_to_color(temp) -> ti.types.vector(3, dtype=ti.f32):
    lookup_value = (temp - 1000.0) / 100.0  # The LUT starts at 1000K and has a step of 100K per index
    lookup_value = tm.clamp(lookup_value, 0.0, color_linear_gpu.shape[0] - 1.001)  # prevent out-of-bounds
    index = ti.cast(ti.floor(lookup_value), ti.i32)
    interpolant = lookup_value - index

    # ensure index+1 is in bounds. should be redundant due to the previous clamp, but just in case
    index = ti.min(index, color_linear_gpu.shape[0] - 2)

    lower_color = color_linear_gpu[index]
    upper_color = color_linear_gpu[index + 1]

    color = tm.mix(lower_color, upper_color, interpolant)

    return color


@ti.func
def temp_to_intensity(temp, t_ref=12000.0, t_vis=4000.0):
    norm = temp / t_ref
    
    # exponential suppression of cool temperatures
    vis_factor = 1.0 - tm.exp(-temp / t_vis)
    
    return tm.pow(norm, 4.0) * vis_factor

@ti.func
def random_gaussian() -> tm.vec2:
    # Borrowed from chapter 1 of https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html
    u1 = max(1e-38, ti.random())
    u2 = ti.random()
    r = tm.sqrt(-2.0 * tm.log(u1))
    theta = 2 * tm.pi * u2
    return r * tm.vec2(tm.cos(theta), tm.sin(theta))


@ti.func
def start_ray(x: int, y: int) -> tm.vec3:
    u, v, w = get_camera_basis()
    aspect = RESOLUTION[0] / RESOLUTION[1]
    half_height = ti.tan(fov[None] / 2.0)
    half_width = half_height * aspect

    jitter = random_gaussian() * 0.375

    pixel_u = (x + 0.5 + jitter.x) / RESOLUTION[0]
    pixel_v = (y + 0.5 + jitter.y) / RESOLUTION[1]

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
    light=ti.types.vector(3, dtype=ti.f32),
    final_transmittance=ti.f32,
    hit_photon_sphere=ti.i32,
    e_r=ti.types.vector(3, dtype=ti.f32),
    e_t=ti.types.vector(3, dtype=ti.f32)
)


@ti.func
def perform_integration(u_0, v_0, max_dphi, ds_target, range_limit, max_steps, e_r, e_t) -> IntegrationResult:
    u, v = u_0, v_0
    phi = 0.0
    inv_photon_sphere = 1.0 / (1.5 * R_S)
    inv_range_limit = 1.0 / (range_limit * R_S)
    hit_photon_sphere = 0
    transmittance = 1.0
    light = tm.vec3(0.0, 0.0, 0.0)

    # initial position
    prev_coords_3d = (1.0 / u) * (e_r * tm.cos(phi) + e_t * tm.sin(phi))

    for i in range(max_steps):
        # choose angle step so arc length ~ ds_target: dphi = ds_target / r = ds_target * u
        dphi_local = ds_target * u
        
        # clamp for stability
        dphi_local = tm.clamp(dphi_local, 1e-5, max_dphi)

        # integrate by adaptive angle
        u_next, v_next = integrate_rk4(u, v, dphi_local)

        phi_next = phi + dphi_local
        coords_3d = (1.0 / u_next) * (e_r * tm.cos(phi_next) + e_t * tm.sin(phi_next))

        delta_3d = coords_3d - prev_coords_3d
        ds = tm.length(delta_3d)

        # sample density with smoothed edges (see next fix)
        height = 0.25
        sigma_a = accretion_density(coords_3d, height)
        sigma_s = sigma_a  # assume that scattering coefficient is the same as absorption for simplicity
        sigma_t = sigma_a + sigma_s

        if sigma_a > 0.0:
            # Redshift / doppler shift calculation
            velocity = orbital_velocity(1.0 / u_next)
            vel_dir = orbital_velocity_direction(coords_3d)
            beta = tm.clamp(velocity / scaled_C, 0.0, 0.99999)
            observer_dir = tm.normalize(-delta_3d)  # due to curved spacetime this is *not* camera_pos - coords_3d
            doppler_cos_theta = vel_dir.dot(observer_dir)  # for doppler shift
            gamma = 1.0 / tm.sqrt(1.0 - beta ** 2)
            g_doppler = 1 / (gamma * (1.0 - beta * doppler_cos_theta))
            g_grav = tm.sqrt(1.0 - R_S * u_next)  # gravitational redshift factor
            g = g_doppler * g_grav
            
            # Temperature calculation
            radius_2d = tm.sqrt(coords_3d.x ** 2 + coords_3d.z ** 2)
            temp_emitted = disk_temperature(radius_2d)
            temp_observed = temp_emitted * g  # Include redshift in temperature for color and brightness calculation
            
            # Color and brightness calculation
            rgb = temp_to_color(temp_observed)
            brightness = temp_to_intensity(temp_observed)
            
            # Apply a vertical falloff to simulate the disk's finite thickness using a Gaussian function
            sigma = height / 3.0
            y_falloff = tm.exp(-0.5 * (abs(coords_3d.y) / sigma) ** 2)
            
            s = -tm.log(1.0 - ti.random()) / sigma_t
            
            if s >= ds:
                # no interaction inside this step
                light += transmittance * ds * rgb * brightness * y_falloff
                transmittance *= tm.exp(-sigma_t * ds)
            else:
                # interaction occurs at distance s along the step
                frac = s / ds
                interact_pos = prev_coords_3d + delta_3d * frac
                light += transmittance * s * rgb * brightness * y_falloff
                transmittance *= tm.exp(-sigma_t * s)
                
                # decide absorption vs scattering
                # decide absorption vs scattering
                if ti.random() < sigma_a / sigma_t:
                    # scattering event
                    new_dir = tm.vec3(ti.random() * 2 - 1, ti.random() * 2 - 1, ti.random() * 2 - 1).normalized()
                    
                    # 1. Recalculate the orbital plane basis vectors
                    r_norm = tm.length(interact_pos)
                    normal = tm.cross(interact_pos, new_dir).normalized()
                    e_r = interact_pos / r_norm
                    e_t = tm.cross(normal, e_r).normalized()
                    
                    # 2. Update integration parameters for the new trajectory
                    u_next = 1.0 / r_norm
                    phi_next = 0.0
                    
                    # Add a tiny epsilon to avoid division by zero if direction is purely radial
                    dir_dot_et = tm.dot(new_dir, e_t)
                    if abs(dir_dot_et) < 1e-7:
                        dir_dot_et = 1e-7 if dir_dot_et >= 0 else -1e-7
                    
                    v_next = -tm.dot(new_dir, e_r) / (r_norm * dir_dot_et)
                    
                    # 3. Override coords_3d so that prev_coords_3d becomes interact_pos at the end of the loop
                    coords_3d = interact_pos
                
                else:
                    # absorption event, photon terminates
                    transmittance = 0.0
                
                
                
        prev_coords_3d = coords_3d
        u, v, phi = u_next, v_next, phi_next

        if transmittance < 1e-3:
            break
        if u > inv_photon_sphere:
            hit_photon_sphere = 1
            break

        moving_away = tm.dot(delta_3d, coords_3d) > 0.0
        if u < inv_range_limit and moving_away:
            break

    return IntegrationResult(u, v, phi, light, transmittance, hit_photon_sphere, e_r, e_t)


@ti.func
def tonemap_aces(color):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return tm.clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0)


@ti.func
def tonemap_agx_eotf(color: tm.vec3):
    # Define 3x3 matrix in taichi
    agx_mat_inv = tm.mat3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116
    )

    val = agx_mat_inv @ color

    # sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    # NOTE: We're linearizing the output here. Comment/adjust when *not* using a sRGB render target
    val = tm.pow(val, tm.vec3(2.2))

    return val


@ti.kernel
def render(frame_idx: ti.i32):
    for x, y in ti.ndrange(RESOLUTION[0], RESOLUTION[1]):
        ray_origin = camera_pos[None]
        ray_dir = start_ray(x, y)
        
        normal = tm.normalize(tm.cross(ray_origin, ray_dir))
        
        e_r = tm.normalize(ray_origin)
        e_t = tm.normalize(tm.cross(normal, e_r))
        
        u_0 = 1.0 / tm.length(ray_origin)
        v_0 = -tm.dot(ray_dir, e_r) / (tm.length(ray_origin) * tm.dot(ray_dir, e_t))
        
        # Keyword arguments are not supported in Taichi device functions
        result = perform_integration(u_0, v_0, 0.01,  0.05, 60, 5000, e_r, e_t)
        
        u_final = result.u
        v_final = result.v
        phi_final = result.phi
        
        cos_phi_final = tm.cos(phi_final)
        sin_phi_final = tm.sin(phi_final)
        
        final_pos_3d = (1.0 / u_final) * (result.e_r * cos_phi_final + result.e_t * sin_phi_final)
        
        v_term = (-v_final / u_final ** 2) * (result.e_r * cos_phi_final + result.e_t * sin_phi_final)
        u_term = (1.0 / u_final) * (-result.e_r * sin_phi_final + result.e_t * cos_phi_final)
        final_dir_3d = tm.normalize(v_term + u_term)
        
        skybox_color = sample_spheremap(final_dir_3d) if result.hit_photon_sphere == 0 else tm.vec3(0.0, 0.0, 0.0)
        
        final_color = result.light * brightness_multiplier + skybox_color * result.final_transmittance

        if frame_idx == 0:
            linear_image[x, y] = final_color
        else:
            prev_color = linear_image[x, y]
            new_color = (prev_color * frame_idx + final_color) / (frame_idx + 1)
            linear_image[x, y] = new_color

@ti.kernel
def init():
    # Camera positions
    
    # Space Telescope
    camera_pos[None] = tm.vec3(80.0, 12.5, 0.0)
    look_at[None] = tm.vec3(0.0, 0.0, 0.0)
    fov[None] = tm.radians(25.0)
    
    # Space telescope, top-down
    # camera_pos[None] = tm.vec3(0.0, 80, 0.0)
    # look_at[None] = tm.vec3(0.0, 0.0, 0.0)
    # fov[None] = tm.radians(25.0)
    
    # Perfectly from side, up-close, wide angle
    # camera_pos[None] = tm.vec3(30.0, 0.0, 0.0)
    # look_at[None] = tm.vec3(0.0, 0.0, 0.0)
    # fov[None] = tm.radians(90.0)


@ti.kernel
def tonemap():
    for x, y in ti.ndrange(RESOLUTION[0], RESOLUTION[1]):
        color = linear_image[x, y]
        srgb_image[x, y] = tonemap_aces(color)


def main():
    init()

    window = ti.ui.Window("Black Hole Ray Tracing", RESOLUTION)
    canvas = window.get_canvas()

    frame_idx = 0
    while window.running:
        render(frame_idx)
        tonemap()

        canvas.set_image(srgb_image)
        window.show()
        frame_idx += 1


if __name__ == "__main__":
    main()
