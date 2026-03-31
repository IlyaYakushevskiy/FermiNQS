# FermiNQS
Code repository for my Thesis "Efficient neural network wave functions for interacting fermions"


### Run the code 
```
python -m venv .venv
source .venv/bin/activate
pip install -r "requirements.txt" 

#if using wanb
wandb login

python main.py +experiment=qho_bosons_gaussian

```
to tweak hyper-parameters one should navigate to configs/experiment/ and either set up a new config (it just over-rides deafult paramaters selectively) or modify existing (e.g. qho_bosons_gaussian.yaml) 

