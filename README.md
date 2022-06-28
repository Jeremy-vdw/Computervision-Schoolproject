# project_computervisie
# Setup
1) Setup the database:

The Mongo database should be seeded when running the docker compose file. 
Run the following command from the workdirectory folder:

```
docker-compose up --detach
```

2) Run the program:

The main file to run is main_multithreding.py.

Run the following command again from the workdirectory folder:

```
python3 painting_recognition/main_multitreading.py
```
Next the database will be fetched and other parameters will be set up, this might take a while.
After this is done a popup will appear where a .avi or .mp4 file can be chosen and the programm will start.

3) Controls:

Next are some possible key functions while the programm is running:

p: pause -> The video and programm will pause.

r: restart -> The programm will restart and let you choose another input video.

q: quit -> The programm will halt.