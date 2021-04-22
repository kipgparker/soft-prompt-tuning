import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self, 
                embedding_dimension: int,
                num_tokens: int = 20, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """[summary]

        Args:
            embedding_dimension (int): [description]
            num_tokens (int, optional): [description]. Defaults to 20.
            random_range (float, optional): [description]. Defaults to 0.5.
        """
        super(Net, self).__init__()
        self.initialize_from_vocab = initialize_from_vocab
        self.wte = None
        self.embedding = nn.parameter.Parameter(torch.FloatTensor(embedding_dimension, num_tokens).uniform_(-random_range, random_range))

    def initialize_embedding(self):
        """Initalizes soft embedding to random values from input WTE emedding
        """
        pass

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """[summary]

        Args:
            new_embeddings (nn.Embedding): [existing WTE embedding within the language model]
        """
        self.wte = new_embeddings
        if self.initialize_from_vocab:
            initialize_embedding()
