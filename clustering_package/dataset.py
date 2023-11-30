import numpy as np
import torch
from torch.utils.data import Dataset

from clustering_package import embedding
from wiki_package.wiki_corpora import WikiCorpus


class DKMDataset(Dataset):
    def __init__(self, dataset_name, language_1='en', language_2='fr', dataset_path=None,
                 embedding_type='', embedding_path=None, get_type='wot', seed=None, emb_kwargs=None):

        self.dataset_name = dataset_name
        self.embedding_type = embedding_type
        self.embedding_path = embedding_path
        self.seed = seed
        self.data, self.target, self.n_clusters, self.mask = None, None, None, None
        self.data_source,  self.embedding_size = {}, None

        if dataset_name[:4] == 'WIKI':
            wiki_corpus = WikiCorpus(corpus_id=dataset_name[5:], language_1=language_1, language_2=language_2,
                                     info_path=dataset_path, load_label_info=False, set_cluster_info=False)
            self.dataset_name = dataset_name[5:]
            self.data_source['dataset'] = wiki_corpus.dataset
            self.data_source['languages'] = [language_1, language_2]
            self.data = None
            self.target = wiki_corpus.target
            self.n_clusters = wiki_corpus.n_clusters
            self.mask = torch.unsqueeze(torch.tensor(wiki_corpus.lang_mask), dim=1)
        else:
            print('Unknown dataset!')

        self.set_embedding_representation(emb_kwargs)

        self.get_item = {
            'wt': self.get_item_with_target,
            'wot': self.get_item_without_target
        }[get_type]

    def set_embedding_representation(self, emb_kwargs):
        if self.embedding_type == 'st_pmL':
            emb_model = embedding.STEmbedding(st_name=self.embedding_type, save_path=self.embedding_path, load_model=False)
            self.data = emb_model.compute(dataset_name=self.dataset_name,
                                          dataset_source=np.array([doc_info['text'] for doc_info in self.data_source]))
        elif self.embedding_type in ['emb_stpm', 'emb_st']:
            info = {'language_1': self.data_source['languages'][0],
                    self.data_source['languages'][0] : {'text':[], 'id': []},
                    'language_2': self.data_source['languages'][1],
                    self.data_source['languages'][1]: {'text':[], 'id': []},
                    'id_order': {self.data_source['languages'][0]:{}, self.data_source['languages'][1]: {}}}
            for i, doc_info in enumerate( self.data_source['dataset']):
                info[self.data_source['languages'][self.mask[i][0].item()]]['text'].append(doc_info['text'])
                info[self.data_source['languages'][self.mask[i][0].item()]]['id'].append(doc_info['id'])
                info['id_order'][self.data_source['languages'][self.mask[i][0].item()]][doc_info['id']] = i

            emb_model = embedding.STEmbeddingLang(st_name=self.embedding_type, save_path=self.embedding_path,
                                                  load_model=False)
            self.data = emb_model.compute(dataset_name=self.dataset_name,
                                          dataset_source=info)
        elif 'ae' in self.embedding_type:
            emb_model = embedding.AEEmbedding(version=self.embedding_type[3:], save_path=self.embedding_path)
            self.data = emb_model.compute(dataset_name=self.dataset_name, seed=self.seed, **emb_kwargs)
        self.embedding_size = self.data.shape[1]

    def update_data(self, new_data):
        self.data = new_data
        self.embedding_size = new_data.shape[1]

    def update_embedding_by_seed(self, seed):
        self.seed = seed
        self.set_embedding_representation()

    def get_item_with_target(self, idx):
        return self.data[idx], self.target[idx], self.mask[idx], idx

    def get_item_without_target(self, idx):
        return self.data[idx], self.mask[idx], idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_item(idx)

