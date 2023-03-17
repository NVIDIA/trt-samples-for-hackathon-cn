#

## Steps to run (in my environment)

1. create docker container

```shell
docker run --gpus '0' -it --name trt-8.5 -p 80:20 -v /home/wili/work:/work -v nvcr.io/nvidia/tensorrt:22.09-py3 /bin/bash
```

2. configure Jupyter Notebook

```shell
# inside the container
python3 -c "from notebook.auth import passwd; print(passwd(algorithm='sha1'))"
# input passwd twice and save the output (for example: sha1:000000000000:0000000000000000000000000000000000000000)
jupyter-notebook --generate-config
# get configuration file /root/.jupyter/jupyter_notebook_config.py
vim /root/.jupyter/jupyter_notebook_config.py
# edit the file as below
"""
c.NotebookApp.ip = '*'                                                                  # keep "*"
c.NotebookApp.port = 20                                                                 # use the second parameter of the option "-p" in "docker run"
c.NotebookApp.open_browser = False                                                      # change to "False"
c.NotebookApp.allow_remote_access = True                                                # change to "True"
c.NotebookApp.password = u'sha1:000000000000:0000000000000000000000000000000000000000'  # use the hash we get before
"""
# start jupyter notebook
jupyter notebook --allow-root

# inside the host

docker ps -a | grep trt-8.4
# get ID of the container (for example: 000000000001
docker inspect 000000000001 | grep IPAddress
# get IP address of the container (for example: 172.17.0.2)

# input the address 172.17.0.2:20 in the browser (using the IP address and port number)
# if configuration is correct, we can input he passed set before and start the jupyter notebook
```

3. install libraries

```shell
chmod +x build.sh
./build.sh
```

4. run the example code

```shell
python3 getOnnxModel.py
python3 mainProcess.py
```

5. open ./model.ipynb in the jupyter notebook of host, we can run the commands line by line
