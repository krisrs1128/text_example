This is a minimal project on text classification, which can be run on the mila
cluster. To go ahead and launch a cluster job running this experiment, go to the
`cluster` folder and type `sbatch text.sbatch`.

A brief description of what's going on here,

- cluster: Wrapper scripts that launch the runs on the Mila cluster.
- config.json: This is a json defining hyperparameters for any particular
  experiment. You can go in and see what's happening.
- doc: More explanations about what's going on in this repo.
- exper.py: The script that choreographs all the data and modeling components.
- exploratory: Parts of the pipeline that visualize raw data and model output.
- pipeline: Code to define the model(s) and perform training.
- data_prep: Utilities for downloading and building vocabulary objects from the
  raw data.
