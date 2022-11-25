#

## 使用步骤（以我的设置为例）
1. 创建 docker container，关键是设置端口映射
```shell
docker run --gpus '0' -it --name trt-8.5 -p 80:20 -v /home/wili/work:/work -v nvcr.io/nvidia/tensorrt:22.09-py3 /bin/bash
```

2. 配置 Jupyter Notebook 以便本地可以打开
+ docker 内运行 `python3 -c "from notebook.auth import passwd; print(passwd(algorithm='sha1'))"`，输入两遍自定义的密码，保存其返回结果（举例 sha1:000000000000:0000000000000000000000000000000000000000）
+ 运行 `jupyter-notebook --generate-config`，产生配置文件 /root/.jupyter/jupyter_notebook_config.py
+ 打开该配置文件，在末尾添加一些信息：
```python
c.NotebookApp.ip = '*'                                                                  # keep "*"
c.NotebookApp.port = 20                                                                 # use the second parameter of the option "-p" in "docker run"
c.NotebookApp.open_browser = False                                                      # change to "False"
c.NotebookApp.allow_remote_access = True                                                # change to "True"
c.NotebookApp.password = u'sha1:000000000000:0000000000000000000000000000000000000000'  # use the hash we get before
```
+ 运行 `jupyter notebook --allow-root`，启动 Jupyter Notebook
+ 在主机中运行 `docker ps -a | grep trt-8.4` 获取 container ID 号（举例 000000000001），运行 `docker inspect 000000000001 |grep IPAddress` 获取 docker IP 地址（举例 172.17.0.2）
+ 在浏览器中输入 `172.17.0.2:20`（分别为上面获得的 docker IP 地址和先前 docker 映射的端口号）
+ 如果配置正确的话，可以在浏览器页面上输入先前设置的密码，进入 Jupyter Notebook

2. 安装相关库
```shell
chmod +x build.sh
./build.sh
```

3. 运行范例程序
```
python3 getOnnxModel.py
python3 mainProcess.py
```

4. 在主机 Jupyter Notebook 中打开 ./model.ipynb，逐命令执行
