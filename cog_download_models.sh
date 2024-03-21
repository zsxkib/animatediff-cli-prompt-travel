#!/bin/bash

mkdir -p data/models/sd/

pget https://civitai.com/api/download/models/78775 data/models/sd/toonyou_beta3.safetensors || true
pget https://civitai.com/api/download/models/72396 data/models/sd/lyriel_v16.safetensors || true
pget https://civitai.com/api/download/models/71009 data/models/sd/rcnzCartoon3d_v10.safetensors || true
pget https://civitai.com/api/download/models/79068 data/models/sd/majicmixRealistic_v5Preview.safetensors || true
pget https://civitai.com/api/download/models/29460 data/models/sd/realisticVisionV40_v20Novae.safetensors || true

# Download Motion_Module models
wget -O data/models/motion-module/mm_sd_v14.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt || true
wget -O data/models/motion-module/mm_sd_v15.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt || true
wget -O data/models/motion-module/mm_sd_v15_v2.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt || true
