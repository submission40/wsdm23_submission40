# wsdm23_submission40
WIP
Code to reproduce experiments described in the article submitted at WSDM 2023 will be available here.
RTA models can be trained using main.py, however code to transform original Million Playlist Dataset into the right format is missing for the moment.

TODO :
- share code to process MPD
- share code for baselines used
- share code to compute all metrics and plot graphs

STEP 1 :
- download MPD https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files#:~:text=spotify_million_playlist_dataset.zip%20(5.39%20GB)
unzip all files in ressources/data/raw_MPD
- run formatting script for rta_models

```
python src/preprocessing/format_rta_input.py ressources/data/raw_MPD/ ressources/data/rta_input
```
- run formatting script for baseline models
- separate train / validation / test

STEP 2 :
- train RTA models
- train baseline models

Optionnal : 
models on validation set

STEP 3:
- compute metrics on test set
- plot figures
