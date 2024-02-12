import os.path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel

from clustering_package.constatnts import EMBEDDING_MODEL_INFO, EMB_PATH
from clustering_package.util_files import utils

class STEmbedding():
    def __init__(self, st_name, save_path=None, load_model=False):
        self.st_model_name = EMBEDDING_MODEL_INFO[st_name]['st_model_name']
        self.normalize = EMBEDDING_MODEL_INFO[st_name]['normalize']
        if save_path is None:
            self.save_path = os.path.join(EMB_PATH, st_name)
        else:
            self.save_path = os.path.join(save_path, st_name)
        self.tokenizer = None
        self.model = None

        if load_model:
            self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{self.st_model_name}')
        self.model = AutoModel.from_pretrained(f'sentence-transformers/{self.st_model_name}')

    def text2embedding(self, texts):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)
        err = 0
        documents_embedding = []
        for document in tqdm(texts):
            try:
                sentences = sent_tokenize(document)
                encoded_input = self.tokenizer(sentences, padding=True, truncation=True,
                                          return_tensors='pt')  # Tokenize sentences
                with torch.no_grad():  # Compute token embeddings
                    model_output = self.model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])  # Perform pooling
                document_embedding = torch.sum(sentence_embeddings, 0)
            except:
                document_embedding = torch.zeros((384,))
                err += 1
            documents_embedding.append(document_embedding)
        documents_embedding = torch.stack(documents_embedding)

        if self.normalize:
            documents_embedding = F.normalize(documents_embedding, p=2, dim=1)

        if err > 0:
            print(f'fail {err} times')

        return documents_embedding

    def load_embedding(self, dataset_name):
        return utils.read_data(os.path.join(self.save_path,
                                            f'dataset_{dataset_name}/embedding_{dataset_name}.tz'))

    def save_embedding(self, dataset_name, documents_embedding):
        utils.save_data(documents_embedding, os.path.join(self.save_path,
                                            f'dataset_{dataset_name}/embedding_{dataset_name}.tz'))

    def compute(self, dataset_name, dataset_source):
        try:
            dataset_embedding = self.load_embedding(dataset_name=dataset_name)
        except FileNotFoundError:
            print('Start embedding computations ....')
            dataset_embedding = self.text2embedding(dataset_source)
            self.save_embedding(dataset_name=dataset_name, documents_embedding=dataset_embedding)
            print('Embedding calculation complete.')
        return dataset_embedding


class STEmbeddingLang():
    def __init__(self, st_name, save_path=None, load_model=False):
        self.st_model_name = EMBEDDING_MODEL_INFO[st_name]['st_model_name']
        self.normalize = EMBEDDING_MODEL_INFO[st_name]['normalize']
        if save_path is None:
            self.save_path = os.path.join(EMB_PATH, st_name)
        else:
            self.save_path = os.path.join(save_path, st_name)
        self.tokenizer = None
        self.model = None

        if load_model:
            self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{self.st_model_name}')
        self.model = AutoModel.from_pretrained(f'sentence-transformers/{self.st_model_name}')

    def text2embedding(self, texts):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)
        err = 0
        documents_embedding = []
        for document in tqdm(texts):
            try:
                sentences = sent_tokenize(document)
                encoded_input = self.tokenizer(sentences, padding=True, truncation=True,
                                          return_tensors='pt')  # Tokenize sentences
                with torch.no_grad():  # Compute token embeddings
                    model_output = self.model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])  # Perform pooling
                document_embedding = torch.sum(sentence_embeddings, 0)
            except:
                document_embedding = torch.zeros((384,))
                err += 1
            documents_embedding.append(document_embedding)
        documents_embedding = torch.stack(documents_embedding)

        if self.normalize:
            documents_embedding = F.normalize(documents_embedding, p=2, dim=1)

        if err > 0:
            print(f'fail {err} times')

        return documents_embedding

    def load_embedding(self, dataset_name, language):
        dataset_id = utils.read_data(os.path.join(self.save_path, f'dataset_{dataset_name}/id_{language}.json'))
        dataset_embedding = utils.read_data(os.path.join(self.save_path, f'dataset_{dataset_name}/embedding_{language}.npy'), data_type='np')
        return dataset_id, dataset_embedding

    def save_embedding(self, dataset_name, document_embeddings, document_ids, language):
        utils.path_check(path=os.path.join(self.save_path, f'dataset_{dataset_name}/'), if_create=True)
        utils.save_data(document_embeddings.detach().numpy(), os.path.join(self.save_path,
                                                                f'dataset_{dataset_name}/embedding_{language}.npy'),
                        data_type='np')
        utils.save_data(document_ids, os.path.join(self.save_path,
                                            f'dataset_{dataset_name}/id_{language}.json'))


    def compute(self, dataset_name, dataset_source):
        n = (len(dataset_source['id_order'][dataset_source['language_1']]) + len(dataset_source['id_order'][dataset_source['language_2']]))
        dataset_embedding = [None] * n
        for language in [dataset_source['language_1'], dataset_source['language_2']]:
            try:
                doc_ids, doc_embs = self.load_embedding(dataset_name=dataset_name, language=language)
            except FileNotFoundError:
                print('Start embedding computations ....')
                documents = dataset_source[language]['text']
                doc_ids = dataset_source[language]['id']
                doc_embs = self.text2embedding(documents)
                self.save_embedding(dataset_name=dataset_name, document_embeddings=doc_embs,
                                    document_ids=doc_ids, language=language)
                doc_embs = doc_embs.detach().numpy()
            for doc_id, doc_emb in zip(doc_ids, doc_embs):
                if doc_id in dataset_source['id_order'][language]:
                    dataset_embedding[dataset_source['id_order'][language][doc_id]] = doc_emb.tolist()
        return np.array(dataset_embedding)


class AEEmbedding():
    def __init__(self, version, save_path=None,):
        self.version = version
        if save_path is None:
            self.save_path = os.path.join(EMB_PATH, 'AE')
        else:
            self.save_path = os.path.join(save_path, 'AE')
        self.model = None

    def load_embedding(self, dataset_name, seed):
        embeddings = utils.read_data(os.path.join(self.save_path,
                                                  f'dataset_{dataset_name}/embedding_ae{self.version}_s{seed}.npy'),
                                     data_type='np')
        return embeddings

    def save_embedding(self,  dataset_name, documents_embedding, seed):
        utils.save_data(documents_embedding,
                        os.path.join(self.save_path, f'dataset_{dataset_name}/embedding_ae{self.version}_s{seed}.npy'),
                        data_type='np')

    def run_ae(self, dataset_name, seed, **kwargs):
        from clustering_package import run
        _, pre_ae_arg = run.run_util.version_verification(alg='AE', alg_info=None, input_version=self.version,
                                                          mode_='read', last_print=True,
                                                          verbose_level=kwargs.get('verbose_level', 1))
        if pre_ae_arg is None:
            pre_ae_arg = run.run_util.generate_version_info(alg='AE', **kwargs)
            ae_ver, pre_ae_arg = run.run_util.version_verification('AE', alg_info=pre_ae_arg, input_version=self.version,
                                                           mode_='create', verbose_level=kwargs.get('verbose_level', 1))
            self.version = ae_ver
        kwargs.update(pre_ae_arg)
        kwargs.update({'embedding_type': pre_ae_arg['source'], 'ae_ver': self.version, 'ver_mode': 'read',
                       'embedding_path': os.path.join(self.save_path, '../'),})
        run.run_autoencoder(f'WIKI_{dataset_name}', seed, **kwargs)

    def compute(self, dataset_name, seed, **kwargs):
        try:
            dataset_embedding = self.load_embedding(dataset_name=dataset_name, seed=seed)
        except FileNotFoundError:
            self.run_ae(dataset_name, seed, **kwargs)
            dataset_embedding = self.load_embedding(dataset_name=dataset_name, seed=seed)
        return dataset_embedding
