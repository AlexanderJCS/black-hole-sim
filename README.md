# black-hole-sim

Derivation of v₀ from Cartesian Direction
In polar coordinates, the position is $(r, \phi)$, so $u = 1/r$.
The chain rule gives us: $v = \frac{du}{d\phi} = \frac{du/dt}{d\phi/dt}$
We need to express $\dot{u}$ and $\dot{\phi}$ in terms of the Cartesian direction $\hat{d} = (d_x, d_y)$.
Finding $\dot{r}$:
$\dot{r} = \frac{d}{dt}|\vec{r}| = \hat{r} \cdot \vec{\dot{r}} = d_x \cos\phi + d_y \sin\phi$
At $\phi = 0$: $\dot{r} = d_x$
Finding $\dot{\phi}$:
The tangential component of velocity is $r\dot{\phi}$, so:
$r\dot{\phi} = -d_x \sin\phi + d_y \cos\phi$
At $\phi = 0$: $r\dot{\phi} = d_y$, so $\dot{\phi} = d_y / r$
Combining:
$v = \frac{du}{d\phi} = \frac{-\dot{r}/r^2}{\dot{\phi}} = \frac{-d_x / r^2}{d_y / r} = -\frac{d_x}{r \cdot d_y} = -u_0 \cdot \frac{d_x}{d_y}$
So the formula v_0 = -u_0 * d_0[0] / d_0[1] is correct, assuming the photon starts on the positive x-axis ($\phi = 0$), which is the implicit assumption here.