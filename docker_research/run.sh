docker run -it \
    --user ${USER}:${GROUP} \
    --shm-size 256G \
    --log-driver=none \
    --gpus all \
    -v $(realpath /home/arseniy-zemerov):/home/${USER}/Research \
    -v $(realpath /home/arseniy-zemerov/data):/data \
    -v $(realpath /storage):/storage \
    --name ${USER}_base ${USER}_base:latest \
    /bin/bash
