import tensorflow as tf
from constants import set_run_session, EXTENSION


# Main specific task
def main_specific_tasks(extension, filename=None):
    # Generate unique identifier to avoid data loss
    if filename is not None:
        set_run_session(extension, filename)
    else:
        set_run_session(extension)

    print("TensorFlow version: ", tf.__version__)

    # Test GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError("GPU device not found")
    print("Found GPU at: {}".format(device_name))
