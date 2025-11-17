import tensorflow as tf
from tensorflow.experimental import dtensor

def send_to_devices(tens):
    # Define a mesh of devices (e.g., 2x2 grid of devices)
    mesh = dtensor.create_mesh([("x", 2)], devices=tf.config.list_physical_devices())
    
    
    # Define a layout for a tensor, sharding it across the 'x' dimension
    layout = dtensor.Layout([dtensor.UNSHARDED, "x"], mesh)
    
    # Create a DTensor with the specified layout
    # The first dimension is replicated, the second is sharded across 'x'
    dtensor_sharded = dtensor.copy_to_layout(tens, layout)
    return tensor_sharded
