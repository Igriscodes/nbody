import taichi as ti
import math

ti.init(arch=ti.cuda) 

N = 8000 
NUM_GALAXIES = 5 
DT = 6e-4 
G = 1.0 
EPS = 0.015 
DAMP = 0.999 
V_MAX = 10.0 
STEPS_PER_FRAME = 8

pos   = ti.Vector.field(2, ti.f32, shape=N)
vel   = ti.Vector.field(2, ti.f32, shape=N)
mass  = ti.field(ti.f32, shape=N)
col   = ti.Vector.field(3, ti.f32, shape=N)
pos01 = ti.Vector.field(2, ti.f32, shape=N)

@ti.kernel
def init_galaxies():
    particles_per_galaxy = N // NUM_GALAXIES
    
    for i in range(N):
        mass[i] = 1.0 / N
        galaxy_id = i // particles_per_galaxy
        
        angle = 2.0 * math.pi * (galaxy_id / NUM_GALAXIES)
        dist_from_origin = 0.5
        center = ti.Vector([ti.cos(angle), ti.sin(angle)]) * dist_from_origin
        
        u = ti.random()
        r_angle = 2.0 * math.pi * ti.random()
        r = 0.25 * ti.sqrt(u) 

        bulk_v = ti.Vector([-ti.sin(angle), ti.cos(angle)]) * 0.2
        
        spin_dir = 1.5 if (galaxy_id % 2 == 0) else -1.5
        p = center + ti.Vector([r * ti.cos(r_angle), r * ti.sin(r_angle)])
        rel = p - center
        tang = ti.Vector([-rel.y, rel.x]).normalized()
        
        pos[i] = p
        vel[i] = bulk_v + spin_dir * tang * (rel.norm() + 0.05)

@ti.kernel
def step():
    for i in range(N):
        ai = ti.Vector([0.0, 0.0])
        xi = pos[i]

        for j in range(N):
            if i != j:
                diff = pos[j] - xi
                dist_sq = diff.dot(diff) + EPS**2
                ai += G * mass[j] * diff * (ti.rsqrt(dist_sq)**3)

        v_new = (vel[i] + DT * ai) * DAMP
        
        speed = v_new.norm()
        if speed > V_MAX:
            v_new = v_new.normalized() * V_MAX
            
        x_new = pos[i] + DT * v_new

        for k in ti.static(range(2)):
            if x_new[k] < -1.0 or x_new[k] > 1.0:
                v_new[k] *= -0.5
                x_new[k] = ti.max(-1.0, ti.min(1.0, x_new[k]))

        vel[i] = v_new
        pos[i] = x_new

@ti.kernel
def shade():
    for i in range(N):
        galaxy_id = i // (N // NUM_GALAXIES)
        hue = galaxy_id / NUM_GALAXIES
        
        base_col = ti.Vector([0.5 + 0.5 * ti.cos(hue * 6.28), 
                              0.5 + 0.5 * ti.cos(hue * 6.28 + 2.09), 
                              0.5 + 0.5 * ti.cos(hue * 6.28 + 4.18)])
        
        speed = vel[i].norm()
        brightness = ti.min(1.0, 0.3 * speed + 0.2)
        col[i] = base_col * brightness
        
        pos01[i] = (pos[i] + 1.0) * 0.5

def main():
    init_galaxies()
    window = ti.ui.Window("Universal N-Body Merger", (900, 900))
    canvas = window.get_canvas()

    while window.running:
        for _ in range(STEPS_PER_FRAME):
            step()
        shade()
        canvas.set_background_color((0.0, 0.0, 0.05))
        canvas.circles(pos01, radius=0.002, per_vertex_color=col)
        window.show()

if __name__ == "__main__":
    main()
