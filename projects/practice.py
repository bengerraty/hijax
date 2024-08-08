import jax
import jax.numpy as jnp

print("using jax", jax.__version__)

a = jnp.zeros((2, 5), dtype=jnp.float32)
b = jnp.arange(7)
print(a)
print(b)
print(jax.devices())