For more information see the list of the requirments (You can install them `pip install -r requirements.txt`). 
The `main.py` is the file to call to start the training. 
The code works with `Python3.9` and `Python3.9-Python3.11`. 
``` Bash
# create conda environment
conda create -n masc python==3.9
conda activate masc
pip install torch, tensorflow
```
Note that this code does work with TensorFlow 2+. 
### Paper citation

### How to Run
For Discreate action/observation space "PPO", and for Continuous action/observation space "SAC", you can start the training with No communication environments
```
python BS1P1F.py --env1
python BS1P1F.py --env2
python BS1P1F.py --env3
python BS1P1F.py --env123plus
...
```

For Continuous action/observation space "SAC",  you can start the training with Mixed environments
```
python SAC1P1F-Mixed.py --env1
python SAC1P1F-Mixed.py --env2
python SAC1P1F-Mixed.py --env3
python SAC1P1F-Mixed.py --env123plus
...
```

For Continuous action/observation space "SAC",  you can start the training with Lying environments
```
python SAC1P1F-Lying.py --env1
python SAC1P1F-Lying.py --env2
python SAC1P1F-Lying.py --env3
python SAC1P1F-Lying.py --env123plus
...
```


For Discreate action/observation space "PPO", and for Continuous action/observation space "SAC",  you can start the training with Tust environments
```
python SAC1P1F.py --env7
python SAC1P1F.py --env8
python SAC1P1F.py --env9
python SAC1P1F.py --env789plus
...
```


