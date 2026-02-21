import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

R_S = 1.0  # Ensure this is a float


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


@ti.func
def perform_integration(u_0, v_0, dt, steps):
    u, v = u_0, v_0
    phi = 0.0  # Explicitly track phi
    
    inv_photon_sphere = 1.0 / (1.5 * R_S)
    inv_range_limit = 1.0 / (50.0 * R_S)
    
    for _ in range(steps):
        u, v = integrate_rk4(u, v, dt)
        phi += dt
        
        if u > inv_photon_sphere or u < inv_range_limit:
            break
    
    return u, v, phi


@ti.kernel
def render():
    for i in range(1):
        ray_origin = ti.math.vec3(0.0, 0.0, 15.0)
        ray_dir = tm.normalize(ti.math.vec3(0.0, 0.5, 0.5))
        
        normal = tm.normalize(tm.cross(ray_origin, ray_dir))
        
        e_r = tm.normalize(ray_origin)
        e_t = tm.normalize(tm.cross(normal, e_r))
        
        u_0 = 1.0 / tm.length(ray_origin)
        v_0 = -tm.dot(ray_dir, e_r) / (tm.length(ray_origin) * tm.dot(ray_dir, e_t))
        
        # Keyword arguments are not supported in Taichi device functions
        u_final, v_final, phi_final = perform_integration(u_0, v_0, 0.001, 10000)
        
        print("Final u, v:", u_final, v_final)
        
        final_pos_3d = (1.0 / u_final) * (e_r * tm.cos(phi_final) + e_t * tm.sin(phi_final))
        
        v_term = (-v_final / u_final ** 2) * (e_r * tm.cos(phi_final) + e_t * tm.sin(phi_final))
        u_term = (1.0 / u_final) * (-e_r * tm.sin(phi_final) + e_t * tm.cos(phi_final))
        final_dir_3d = tm.normalize(v_term + u_term)
        
        print("Final Position:", final_pos_3d)
        print("Final Direction:", final_dir_3d)


def main():
    render()


if __name__ == "__main__":
    main()