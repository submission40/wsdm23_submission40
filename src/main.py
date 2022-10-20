import pickle, argparse

from src.data_manager.data_manager import DataManager

from src.rta.utils import get_device
from src.rta.rta_model import RTAModel
from src.rta.aggregator.gru import GRUNet
from src.rta.representer.base_representer import BaseEmbeddingRepresenter

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--model_name", type = str, required = True,
                        help = "Name of model to train")
      args = parser.parse_args()

      data_manager = DataManager()
      with open("best_params.pkl", "rb") as f:
        p = pickle.load(f)

      tr_params = p[args.model_name]

      if args.model_name == "GRU":
        Emodel = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
        gruModel = GRUNet(data_manager.n_tracks + 1, tr_params['d'], tr_params['h_dim'], tr_params['d'], tr_params['n_layers'], tr_params['drop_p'])
        rta_gru = RTAModel(data_manager, Emodel, gruModel, training_params = tr_params).to(get_device())
        #recos = rta_gru.compute_recos(rta_gru, test_dataloader, n_recos)
        rta_gru.run_training(tuning=True)