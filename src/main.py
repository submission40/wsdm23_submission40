import json, argparse

from src.data_manager.data_manager import DataManager

from src.rta.utils import get_device
from src.rta.rta_model import RTAModel
from src.rta.aggregator.gru import GRUNet
from src.rta.aggregator.decoder import DecoderModel
from src.rta.aggregator.base import AggregatorBase
from src.rta.representer.base_representer import BaseEmbeddingRepresenter
from src.rta.representer.fm_representer import FMRepresenter
from src.rta.representer.attention_representer import AttentionFMRepresenter
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type = str, required = True,
                    help = "Name of model to train")
  args = parser.parse_args()

  data_manager = DataManager()
  with open("best_params.json", "r") as f:
    p = json.load(f)

  tr_params = p[args.model_name]
  savePath = "ressources/models/%s" % args.model_name
  if args.model_name == "GRU":
    print("Initialize Embeddings")
    Emodel = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize GRU")
    gruModel = GRUNet(data_manager.n_tracks + 1, tr_params['d'], tr_params['h_dim'], tr_params['d'], tr_params['n_layers'], tr_params['drop_p'])
    rta_model = RTAModel(data_manager, Emodel, gruModel, training_params = tr_params).to(get_device())

  if args.model_name == "AVG":
    print("Initialize Embeddings")
    Emodel = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize GRU")
    aggModel = AggregatorBase()
    rta_model = RTAModel(data_manager, Emodel, aggModel, training_params = tr_params).to(get_device())

  if args.model_name == "Dec":
    print("Initialize Embeddings")
    Emodel = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize Decoder")
    decoderModel = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])
    rta_model = RTAModel(data_manager, Emodel, decoderModel, training_params = tr_params).to(get_device())

  if args.model_name == "Dec-FM":
    print("Initialize Embeddings")
    FMModel = FMRepresenter(data_manager, tr_params['d'])
    print("Initialize Decoder")
    decoderModel = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])
    rta_model = RTAModel(data_manager, FMModel, decoderModel, training_params = tr_params).to(get_device())

  if args.model_name == "Dec-NN":
    print("Initialize Embeddings")
    AttFMModel = AttentionFMRepresenter(data_manager, emb_dim=tr_params['d'], n_att_heads=tr_params['n_att_heads'], n_att_layers=tr_params["n_att_layers"], dropout_att=tr_params["drop_att"])
    print("Initialize Decoder")
    decoderModel = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])
    rta_model = RTAModel(data_manager, AttFMModel, decoderModel, training_params = tr_params).to(get_device())
  print("Train model %s" % args.model_name)
  rta_model.run_training(tuning=True, savePath=savePath)