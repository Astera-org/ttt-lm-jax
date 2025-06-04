import mlxu
from jax.sharding import Mesh
import jax
import numpy as np  


def main(argv):

    rng=jax.random.PRNGKey(0)

    mesh = Mesh( np.array(jax.devices()).reshape((-1,1,1)), ('data','fsdp','mp'))

    rng1,_ = jax.random.split(rng)

    with mesh:
        rng2, _ = jax.random.split(rng1)
        print("after")
    

mlxu.run(main)
