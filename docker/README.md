# Using Docker for CASPER-3D
This folder contains relevant files for using Docker to work with CASPER-3D.

## Building the Docker Image
Run the following command in this folder:
```bash
docker build -t casper .
```
This will create a Docker image tagged `casper:latest`.

## Running the Container
Run the following command:
```bash
docker run --rm -it --entrypoint bash casper:latest
```
This will open a bash shell that lives in a Docker container built from `casper:latest`, which you can use to test the environment.

For development, it will be more useful if we can access the code and datasets inside the container.
Assume the repo is located at `/home/USER/code/CASPER-3D` on your local machine, 
and the dataset is located at `/mnt/USER/datasets/casper-data`.
Run the following command to load datasets and our code repo:
```bash
docker run --rm --gpus all -it --entrypoint bash -v /home/USER/code/CASPER-3D:/opt/project/CASPER-3D  \
-v /mnt/USER/datasets/casper-data:/mnt/datasets/ casper:latest
```
which will 1) launch a container based on `casper:latest`, 2) mount the repo to `/opt/project/CASPER-3D`, 
and 3) mount the dataset at `/mnt/datasets/`. 
Note that all changes to the mounted folder will be reflected on your local disks.

You can then run unit tests by running the following in the newly created container:
```bash
PYTHONPATH="${PYTHONPATH}:/opt/project/CASPER-3D/src" pytest /opt/project/CASPER-3D/tests/ 
```

You can also install the package inside the container:
```bash
cd /opt/project/CASPER-3D
python -m pip install --no-build-isolation -e .
```
The `--no-build-isolation` is used to prevent `pip` from downloading `pytorch` again, as we already have the requirements met in the container.

## Enable matplotlib Visualization
Run the following:
```bash
docker run -it --user=$(id -u $USER):$(id -g $USER) --env="DISPLAY=:1" --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" casper:latest bash
```
Note that you need to change `--env="DISPLAY=:1` to be what you have on your host machine (which you can get by running `echo $DISPLAY`).

## Shared memory error
If you see an error regarding data loader and shared memory when running in docker, try adding `--shm-size 8G` to the container.