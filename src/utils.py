import subprocess
import tensorflow as tf

def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,shell=True) #in py 3.5: universal_newlines=True; in 3.7+ text=True
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass



# Specify the GPU ID of the device you wish to use. 
# Note: The GPU ID starts at 0.
def set_gpu(gpu_id):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        print(e)
    else:
        print("No GPUs available")

