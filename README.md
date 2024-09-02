## Multi-Agent Decision Making in Non-Stationary Supply Chains Environments 

 The code of the paper `Multi-Agent Decision Making in Non-Stationary Supply Chains Environments ` is presented at this repository.

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
For Discreate action/observation space "PPO", you can start the training with
```
python BS1P1F.py --env1
python BS1P1F.py --env2
python BS1P1F.py --env3
...
```

For Continuous action/observation space "SAC", you can start the training with
```
python SAC1P1F.py --env1
python SAC1P1F.py --env2
python SAC1P1F.py --env3
...
```


### BibTeX

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={2024}
}
```
