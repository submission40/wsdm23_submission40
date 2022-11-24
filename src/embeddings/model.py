'''
A module to manage recommendation models.
'''
import numpy as np
import implicit
from math import ceil
from scipy.sparse import csr_matrix, load_npz, save_npz
from functools import reduce
import os
import operator
import random
import torch
from src.data_manager.data_manager import DataManager
class CompletionModel():
  '''
  An abstract class for completion models.
  All subclasses must implement the method score_tracks_for_val_playlists (called when evaluating performance).
  '''
  def __init__(self, data_manager):
    self.data_manager = data_manager
    return
  
  def complete_val_playlists(self, n_recos=500, batch_size=-1, n_seed=5):
    '''
    
    '''
    self.prepare_for_completion(n_seed)
    val_input = self.data_manager.binary_val_sets[n_seed].indices.reshape((self.data_manager.binary_val_sets[n_seed].shape[0],n_seed))
    n_playlists = len(self.data_manager.val_indices)
    if batch_size == -1:
      batch_size = n_playlists
    max_range = ceil(n_playlists / batch_size)
    recos = np.zeros((n_playlists, n_recos + n_seed))
    for i in range(max_range):
      lower_bound = i * batch_size
      upper_bound = min((i+1) * batch_size, n_playlists)
      scores = self.score_tracks_for_val_playlists(range(lower_bound,upper_bound),n_seed = n_seed)
      recos[lower_bound:upper_bound] = np.argsort(-scores)[:, :n_recos + n_seed]
    final_recos = np.zeros((n_playlists, n_recos))
    for i in range(n_playlists):
      final_recos[i] = [j for j in recos[i] if j not in val_input[i]][:n_recos]
    return final_recos
  
  def prepare_for_completion(self, n_seed):
    return

  def score_tracks_for_val_playlists(self, indices, n_seed):
    return


class ChartsModel(CompletionModel):
  # A class that completes every playlists with most popular songs
  def __init__(self, data_manager, name='charts_model'):
    self.data_manager = data_manager
    self.name=name

  def score_tracks_for_val_playlists(self, indices):
    scores = -np.array([self.data_manager.tracks_rank[:-1] for i in indices])
    return scores

class UserUserModel(CompletionModel):
  # A nearest neighbour class that uses similarity between validation playlists 
  # and train playlists to complete playlists (deprecated, replaced with models in baselines modules)
  def __init__(self, data_manager, retrain = False, foldername="data/models/nn", name='user_user_model', beta=0.9):    
    self.data_manager = data_manager
    self.beta = beta
    self.foldername = foldername
    self.name = name

  def prepare_similarities(self, retrain, n_seed=5):
    if retrain :
      val_norms = np.sqrt(np.sum(self.data_manager.binary_val_sets[n_seed] > 0 , axis = 1))
      normed_val_set = self.data_manager.binary_val_sets[n_seed].multiply(csr_matrix(1/val_norms))
      train_norms = np.sqrt(np.sum(self.data_manager.binary_train_set > 0 , axis = 1))
      normed_train_set = self.data_manager.binary_train_set.multiply(csr_matrix(1/train_norms))
      sim = normed_val_set.dot(normed_train_set.T).dot(self.data_manager.binary_train_set)
      if self.beta:
        popularity_count = np.sum(self.data_manager.binary_train_set > 0 , axis = 0)[0]
        popularity_count = np.array(popularity_count)[0]
        popularity_norm = popularity_count / np.max(popularity_count)
        new_pop = popularity_norm **(-(1-self.beta))
        sim = csr_matrix(sim.multiply(new_pop))
      save_npz("%s/similarity_from_user_user_%d" % (self.foldername, n_seed), sim)
    else:
      sim = load_npz("%s/similarity_from_user_user_%d.npz" % (self.foldername, n_seed))
    return sim

  def prepare_for_completion(self, n_seed):
    self.sim = self.prepare_similarities(False, n_seed=n_seed)

  def score_tracks_for_val_playlists(self, indices, n_seed):
    scores = self.sim[indices].toarray()
    return scores

class ItemItemModel(CompletionModel):
  # A nearest neighbour class that uses similarity between songs in the validation set
  # and songs in the train set to complete playlists (deprecated, replaced with models in baselines modules)
  def __init__(self, data_manager, retrain = False, foldername="data/models/nn", name='item_item_model', beta=1.3):    
    self.data_manager = data_manager
    self.beta=beta
    self.foldername = foldername
    self.name=name
  
  def prepare_similarities(self, retrain, n_seed=5):
    if retrain :
      track_norms = np.sqrt(np.sum(self.data_manager.binary_train_set > 0 , axis = 0))
      track_norms[track_norms == 0] = 1
      normed_tracks = self.data_manager.binary_train_set.T.multiply(csr_matrix(1/track_norms).T)
      sim = self.data_manager.binary_val_sets[n_seed].dot(normed_tracks).dot(normed_tracks.T)
      if self.beta:
        popularity_count = np.sum(self.data_manager.binary_train_set > 0 , axis = 0)[0]
        popularity_count = np.array(popularity_count)[0]
        popularity_norm = popularity_count / np.max(popularity_count)
        new_pop = popularity_norm **(-(1-self.beta))
        sim = csr_matrix(sim.multiply(new_pop))
      save_npz("%s/similarity_from_item_item_%d" % (self.foldername, n_seed), sim)
    else:
      sim = load_npz("%s/similarity_from_item_item_%d.npz" % (self.foldername, n_seed))
    return sim

  def prepare_for_completion(self, n_seed):
    self.sim = self.prepare_similarities(False, n_seed=n_seed)

  def score_tracks_for_val_playlists(self, indices, n_seed):
    scores = self.sim[indices].toarray()
    return scores

class MatrixFactorizationModel(CompletionModel):
  # A class that performs matrix factorization of train set, creating embeddings
  # for songs in the train set. Validation playlists are completed by creating an
  # embedding for them that is the average of songs in the playlists, and then finding
  # nearest neighbours in the embedded space
  def __init__(self, data_manager, foldername="ressources/data/embeddings", emb_size= 128, retrain=False, name='mf_model'):
    self.foldername = foldername
    os.makedirs(self.foldername, exist_ok=True)
    self.emb_size = emb_size
    self.data_manager = data_manager
    self.prepare_item_factors(retrain=retrain)
    self.name = name
    
  def prepare_item_factors(self, retrain=False):
    if retrain:
      if torch.cuda.is_available():
        use_gpu = True
      else:
        use_gpu = False
        print("No GPU found, training will be very slow!")
      als_model = implicit.als.AlternatingLeastSquares(factors=self.emb_size, calculate_training_loss=True, iterations=10, regularization=12, use_gpu=use_gpu, use_cg=True, use_native=False, )
      # hyperparameters were tuned with prior Optuna study
      als_model.fit(self.data_manager.binary_train_set)
      # annoying but necessary trick
      np.save('%s/song_embeddings_%d' % (self.foldername, self.emb_size), als_model.item_factors.to_numpy())

      print("if retrain is True a sparse binary train set must be given as input")
      
    else:
      self.item_factors = np.load('%s/song_embeddings_%d.npy' % (self.foldername, self.emb_size))
  
  def build_playlist_vector(self, playlist_tracks):
    count = 0
    playlist_vector = np.zeros((self.emb_size ))
    for j in playlist_tracks:
      playlist_vector += self.item_factors[j]
      count += 1
    playlist_vector = playlist_vector / count
    return playlist_vector

  def build_set_vectors(self, playlist_set):
    n_playlists = playlist_set.shape[0]
    playlist_vectors = np.zeros((n_playlists, self.emb_size))
    for i in range(n_playlists):
      playlist_tracks = playlist_set.indices[playlist_set.indptr[i]:playlist_set.indptr[i+1]]
      playlist_vectors[i] = self.build_playlist_vector(playlist_tracks)
    return playlist_vectors

  def score_tracks_for_val_playlists(self, indices, n_seed):
    playlist_track_subset = self.data_manager.binary_val_sets[n_seed][indices]
    playlist_vectors =  self.build_set_vectors(playlist_track_subset)
    scores = ((self.item_factors).dot(playlist_vectors.T)).T
    return scores

class EnsembleModel(CompletionModel):
  # A class that linearly combines several other models (deprecated, replaced with models in baselines modules)
  def __init__(self, data_manager, models, weights=False, name='ensemble_model'):
    self.data_manager = data_manager
    self.models = models
    if weights:
      self.weights = weights  
    else:
      self.weights = [0.0 for m in models]
    self.name=name

  def iterate_models_scores(self, indices):
    for m in self.models:
      yield m.score_tracks_for_val_playlists(indices)
  
  def score_tracks_for_val_playlists(self, indices):
    print(indices)
    score_iterator = self.iterate_models_scores(indices)
    scores = reduce(operator.add, [el[0] * el[1] for el in zip(self.weights, score_iterator)])
    return scores

  # little algorithm to count the number of possible combinations
  # useful to create grid before gridsearch
  def get_combinations(self, colors, n_elements, use_all = True):
    min_value = int(use_all) # force using every model in the ensemble 
    n_colors = len(colors)
    if n_colors == 2:
      return [[i, n_elements - i] for i in range(min_value, n_elements)]
    elif n_colors < 2:
      print("n_colors must be 2 or higher")
    else :
      comb = []
      for i in range(min_value, n_elements):
        for el in self.get_combinations(colors[1:], n_elements-i, use_all):
          new_el = [i]+el
          comb.append(new_el)
      return comb
  
  def find_optimal_weights(self, evaluator, sample=False, use_all=True, n_recos=500, batch_size=100):
    n_models = len(self.models)
    n_playlists = len(self.data_manager.val_indices)
    combinations = self.get_combinations(self.models, 2*n_models, use_all = use_all)
    if sample:
      combinations = random.sample(combinations, sample)
    n_combinations = len(combinations)
    ndcgs = []
    recos = np.zeros((n_playlists, n_recos, n_combinations))
    max_range = ceil(n_playlists / batch_size)
    for i in range(max_range):
      lower_bound = i * batch_size
      upper_bound = min((i+1) * batch_size, n_playlists)
      current_combination = 0
      for weights in combinations:
        self.weights = weights
        score_iterator = self.iterate_models_scores(range(lower_bound,upper_bound))
        scores = reduce(operator.add, [el[0] * el[1] for el in zip(self.weights, score_iterator)])
        recos[lower_bound:upper_bound,:,current_combination] = np.argsort(-scores)[:, :n_recos]
        current_combination +=1
    for i in range(n_combinations):
      ndcgs.append(evaluator.compute_ndcg(recos[:,:,i]))
    return combinations, recos, ndcgs