# Ising Model

Simple code to viuslize the sampling of ising model configurations around three temperature values: below, at, and above the crtical temperature $T_c$. The app is hosted in [streamlit cloud](https://ising-model.streamlit.app/)

## Run locally

You can use conda to install the dependencies and run the code locally, either by 

```bash
bash> conda env create -f env.yaml
bash> conda activate ising_env 
bash> streamlit run main.py
```

or 
```bash 
bash>conda create -n ising-env 
bash>conda activate ising_env 
bash>conda install --yes --file requirements.txt
bash> streamlit run main.py 
```

## Refrences
Nice refrences 

- [first](https://dpotoyan.github.io/Statmech4ChemBio/4_ising/02_MCMC.html)
- [second](https://github.com/fontclos/stat-mech-python-course/blob/master/notebooks/4-Ising-Model.ipynb)
- [third](https://en.wikipedia.org/wiki/Ising_model)