# Delete all .mp4 files in the current directory and its subdirectories
sudo find . -type f -name '*.mp4' -delete
# Delete all .safetensors files in the current directory and its subdirectories
sudo find . -type f -name '*.safetensors' -delete
# Delete all .ckpt files in the current directory and its subdirectories
sudo find . -type f -name '*.ckpt' -delete
# Delete all .tar files in the current directory and its subdirectories
sudo find . -type f -name '*.tar' -delete
# Delete the stable-diffusion-v1-5 directory
sudo rm -rf data/models/huggingface/stable-diffusion-v1-5
# Delete all .ckpt files in the data/models/motion-module directory
sudo find data/models/motion-module/ -type f -name '*.ckpt' -delete
# Delete all .safetensors files in the data/models/sd directory
sudo find data/models/sd/ -type f -name '*.safetensors' -delete
