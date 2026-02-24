# Black Hole Renderer

Semi-realistic Swarzchild black hole renderer using ray marching and the Schwarzschild metric.

Implemented in the Taichi programming language, which is embedded in Python.

Effects to-add:
* Artistic effects
  * Perlin noise in the accretion disk to add variation (stretched tangentially)
  * Better tonemapping function that can handle the high dynamic range of the scene better
  * Bloom
* Temporal effects
  * Spinning accretion disk
  * Camera movement / video rendering
* Physically-based effects
  * More physically accurate blackbody radiation of the accretion disk (e.g., what temperature is it really?)

Bugs:
* NaNs! (they're not frequent; just filter them)
* Camera farther than 50 units cannot render the black hole

Effects added:
* Gravitational redshift
* Relativistic doppler shift
* Gravitational lensing
* Blackbody radiation of accretion disk
* Volume rendering of accretion disk
* Relativistic beaming
