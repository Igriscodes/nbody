# nbody

High-performance N-Body gravity simulation built with **Taichi Lang**. This project simulates the gravitational interaction and eventual merger of multiple galaxies, utilizing GPU acceleration to handle thousands of particles in real-time.

## Features

* **GPU Accelerated:** Uses the Taichi library to offload heavy physics calculations to the GPU (CUDA/Vulkan/Metal).
* **Multi-Galaxy Setup:** Simulates several distinct star systems with initial orbital velocities and spins.
* **Dynamic Visuals:** Star colors are mapped to their parent galaxy and brightness is tied to their current velocity.
* **Safety Softening:** Implements gravitational softening and velocity clipping to prevent numerical instability and "exploding" particles.

## Prerequisites

Before running the simulation, ensure you have Python installed and the Taichi library:

```bash
pip install taichi
```

*Note: For optimal performance, a CUDA-capable GPU is recommended, though Taichi will automatically fallback to other backends if necessary.*

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Igriscodes/nbody.git
cd nbody
```


2. Run the simulation:
```bash
python app.py
```

## Simulation Mechanics

The core logic follows the standard N-Body gravitational force equation with a softening factor:

$$F = G \frac{m_1 m_2}{r^2 + \epsilon^2}$$

* **$G$:** Gravitational constant.
* **$\epsilon$(EPS):** Softening factor to prevent infinite forces when particles are nearly overlapping.
* **Damping:** A slight velocity damping is applied to maintain system stability over long durations.

## Configuration

You can modify the simulation behavior directly in `app.py` by changing the following constants:

| Variable | Description | Default |
| --- | --- | --- |
| `N` | Total number of particles | 8000 |
| `NUM_GALAXIES` | Number of star systems | 5 |
| `DT` | Time step per calculation | 6e-4 |
| `V_MAX` | Velocity limit for stability | 10.0 |

## License
[GNU Lesser General Public License v2.1](LICENSE) - Feel free to use and modify
