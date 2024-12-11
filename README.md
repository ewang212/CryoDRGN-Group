# CryoDRGN-Group

Implementation of CryoDRGN by students at Caltech. 

This github Repo consists of two models: CryoDRGN VAE (encoder decoder structure cited in our final report) as well as Spatial VAE (our unique rendition of estimating rotation built on top of CryoDRGN's architecture)

Each model has a separate config file. 
To train, run:
```
train_model.py --config_file <file_name>
```

To perform validation, run:
```
validation.py --config_file <file_name>
```

CryoDRGN is the default, and its config file is `config.yaml`, while our two variations of Spatial VAE are `config_spatial_mlp.yaml` and `config_spatial.yaml`. 

The configs are currently set to plot training and validation graphs and data. If unwanted, please change the config file. 

If desired, checkpoints are available to load the models. CryoDRGN VAE has checkpoints `ckpts_000` and `ckpts_001`, corresponding to training on protein 000 and protein 001. 

Spatial VAE has checkpoints `ckpts_spatial` and `ckpts_spatial_mlp`.

Unfortunately, the checkpoints are too large to upload to github, so here is a link: https://drive.google.com/file/d/1tEpwDLfhBfca_NO3XNF_ct85OYEZLXjx/view?usp=sharing

Note: Remember to change directory paths in config file to your local directory.
