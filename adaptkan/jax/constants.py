import jax.numpy as jnp

# Fixed coefficient matrices for B-spline interpolation (orders k=1..5).
M1 = jnp.array([[-1,1],[1,0]], dtype=jnp.float32)
M2 = (1/2)*jnp.array([[1,-2,1],[-2,2,1],[1,0,0]], dtype=jnp.float32)
M3 = (1/12)*jnp.array([[-2,6,-6,2],[6,-12,0,8],[-6,6,6,2],[2,0,0,0]], dtype=jnp.float32)
M4 = (1/24)*jnp.array([[1,-4,6,-4,1],[-4,12,-6,-12,11],[6,-12,-6,12,11],[-4,4,6,4,1],[1,0,0,0,0]], dtype=jnp.float32)
M5 = (1/120)*jnp.array([[-1,5,-10,10,-5,1],[5,-20,20,20,-50,26],[-10,30,0,-60,0,66],[10,-20,-20,20,50,26],[-5,5,10,10,5,1],[1,0,0,0,0,0]], dtype=jnp.float32)