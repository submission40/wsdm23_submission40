# wsdm23_submission40
Code to reproduce experiments described in the article submitted at WSDM 2023 is available here.
RTA models can be trained using main.py, however code to transform original Million Playlist Dataset into the right format is missing for the moment.

## STEP 1 : Download and format dataset
- clone the current project
```
git clone https://github.com/submission40/wsdm23_submission40.git
cd wsdm23_submission40
pip install -R requirements.txt
```
- download MPD https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files#:~:text=spotify_million_playlist_dataset.zip%20(5.39%20GB)
You need to create an account and register to the challenge to be able to download the Million Playlist Dataset
unzip all files in a single folder (for instance : ressources/data/raw_MPD)
Run the script to format the MPD
- run formatting script for rta_models (this should take approximately 1 hour)
```
python src/preprocessing/format_rta_input --mpd_path PATH/TO/UNZIPPED/MPD
```
## STEP 2: Train an RTA model
Example : train a decoder model
```
python -m src.main --model_name Dec
```
Training can be interrupted at any time, an intermdiary model is saved at every epoch

## STEP 3: Evaluate 1 or more RTA models
Example : compute ndcg for a decoder model
```
python -m src.plot_results --metric ndcg --models Dec
```

## (Optionnal) STEP 4: Format and train baselines
Format MPD for all baseline models
```
python -m src.format_baseline_input --mpd_path /PATH/TO/UNZIPPED/MPD
```
Example : train a vsknn model and compute recos

```
python -m run_baselines --model_name vsknn
```

After training, baseline models can be evaluated using the same script as rta model (src.plot_results)
