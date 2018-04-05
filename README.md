# Demo stand. Model: SQuAD (English)

## Installation and start
1. Clone the repo and `cd` to project root:
    ```
    git clone https://github.com/deepmipt/stand_squad_en.git
    cd stand_squad_en
    ```
2. Run script to download and unpack model components:
    ```
    ./download_components.sh
    ```   
3. Create a virtual environment with `Python 3.6`:
    ```
    virtualenv env -p python3.6
    ```
4. Activate the environment:
    ```
    source ./env/bin/activate
    ```
5. Install requirements:
    ```
    pip install -r requirements.txt
    ```
6. Download NLTK data:
    ```
    $ python3
    >>> import nltk
    >>> nltk.download('punkt')
    ```
7. Specify model endpoint host (`api_host`) and port (`api_port`) in `squad_agent_config.json`
8. Specify `CUDA_VISIBLE_DEVICES` and virtual environment path (if necessary) in `run_en_squad.sh`
9. Run model:
    ```
    ./run_en_squad.sh
    ```
## Building and running with Docker:
1. If necessary, build Base Docker image from:

   https://github.com/deepmipt/stand_docker_cuda
  
2. Clone the repo and `cd` to project root:
    ```
    git clone https://github.com/deepmipt/stand_squad_en.git
    cd stand_squad_en
    ```
3. Build Docker image:
   ```
   sudo docker build -t stand/squad_en .
   ```
4. Run Docker image:
   ```
   sudo docker run -p <host_port>:6008 --runtime=nvidia --device=/dev/nvidia<gpu_unit_int_id> -v </path/to/host/vol/map/dir>:/logs stand/squad_en
   ```