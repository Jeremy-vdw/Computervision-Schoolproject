FROM mongo

COPY paintings.json /paintings.json
CMD mongoimport --host mongodb --db project_computervisie --collection paintings --type json --file /paintings.json