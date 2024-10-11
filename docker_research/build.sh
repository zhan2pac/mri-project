docker build \
    -f Dockerfile \
    --build-arg USER=${USER} \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --tag ${USER}_base:latest .