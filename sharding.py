import tensorflow as tf
from tensorflow.experimental import dtensor
import h5py
import jax

def send_one_tensor(tens):
    # Define a mesh of devices (e.g., 2x2 grid of devices)
    mesh = dtensor.create_mesh([("x", 2)], devices=tf.config.list_physical_devices())
    
    
    # Define a layout for a tensor, sharding it across the 'x' dimension
    layout = dtensor.Layout([dtensor.UNSHARDED, "x"], mesh)
    
    # Create a DTensor with the specified layout
    # The first dimension is replicated, the second is sharded across 'x'
    dtensor_sharded = dtensor.copy_to_layout(tens, layout)
    return tensor_sharded

def send_one_tensor_vertically(tens):
    # Define a mesh of devices (e.g., 2x2 grid of devices)
    mesh = dtensor.create_mesh([[("x", 2)]], devices=tf.config.list_physical_devices())
    
    
    # Define a layout for a tensor, sharding it across the 'x' dimension
    layout = dtensor.Layout([dtensor.UNSHARDED, "x"], mesh)
    
    # Create a DTensor with the specified layout
    # The first dimension is replicated, the second is sharded across 'x'
    dtensor_sharded = dtensor.copy_to_layout(tens, layout)
    return tensor_sharded

def send_to_devices(model_name: str):
    weights = 0
    
    def mapping_func(node):
        if 'kernel' in node.name:
            return send_one_tensor_vertically(node)

        else:
            return node

    def reading_state_dict(fpath):
        with h5py.File(fpath, 'r') as f:
            return jax.tree.map(mapping_func, f)

    return reading_state_dict(model_name)


