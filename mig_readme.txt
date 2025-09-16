sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -i 4 -C # CREATE MIG partitions for GPU 4
 
# To reset:
# Destroy ALL compute instances first 
sudo nvidia-smi mig -dci # Then destroy ALL GPU instances   
sudo nvidia-smi mig -dgi