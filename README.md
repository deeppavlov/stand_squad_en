# Demo stand: SQuAD (English)

## Installation and start
1. Create a virtual environment with `Python 3.6`:
    ```
    virtualenv env
    ```
2. Activate the environment:
    ```
    source ./env/bin/activate
    ```
3. Clone the repo and `cd` to project root:
   ```
   git clone https://github.com/deepmipt/stand_squad_en.git
   cd DeepPavlov
   ```
4. Install the requirements:
    ```
    pip install -r requirements.txt
    ```
5. Run script to download and unpack model components:
    ```
    ./download_components.sh
    ```
6. Specify `CUDA_VISIBLE_DEVICES` and virtual environment in `run_en_squad.sh`
7. Run model:
    ```
    ./run_en_squad.sh
    ```