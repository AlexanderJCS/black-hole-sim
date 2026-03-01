# Black Hole Renderer

Semi-realistic Swarzchild black hole renderer using ray marching and the Schwarzschild metric.

Implemented in the Taichi programming language, which is embedded in Python.

Effects to-add:
* Artistic effects
  * Better tonemapping function that can handle the high dynamic range of the scene better
  * Bloom
* Temporal effects
  * Spinning accretion disk
  * Camera movement / video rendering
* Physically-based effects
  * More physically accurate blackbody radiation of the accretion disk (e.g., what temperature is it really?) 
* Optimizations
  * Compute perlin noise as a texture and just sample that texture
* Bugfixes
  * Make the accretion disk texture periodic for theta

Effects added:
* Perlin noise to the accretion disk to provide texture and detail
* Ray tracing the accretion disk
* Gravitational redshift
* Relativistic doppler shift
* Gravitational lensing
* Blackbody radiation of accretion disk
* Volume rendering of accretion disk
* Relativistic beaming
