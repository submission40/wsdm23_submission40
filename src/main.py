import pickle, argparse

from src.gated_cnn import RegressionGatedCNN
from src.evaluator import Evaluator
from src.data_manager import DataManager, EvaluationDataset, SequentialTrainDataset, TransformerTrainDataset, pad_collate, pad_collate_transformer
from src.model import ChartsModel, CompletionModel, MatrixFactorizationModel, ItemItemModel, UserUserModel, EnsembleModel

import implicit
from src.rta.utils import get_device
from src.rta.rta_model import RTAModel
from src.rta.aggregator.base import AggregatorBase
from src.rta.aggregator.decoder import DecoderModel
from src.rta.aggregator.gru import GRUNet
from src.rta.representer.base_representer import BaseEmbeddingRepresenter
from src.rta.representer.attention_representer import AttentionFMRepresenter
from src.rta.representer.fm_representer  import FMRepresenter



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