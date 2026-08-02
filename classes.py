from NGPM import graph_generator
import torch
import torch.nn as nn
class Pattern():
    def __init__(self, non_anon, anon):
        self.semantic_walk=non_anon
        self.anonymous_walk=anon

class PatternEncoder(nn.Module):
    def __init__(self, node_count, embedding_dim):
        super().__init__()
        self.node_count = node_count
        self.embedding_dim = embedding_dim
        self.semantic_embedding = nn.Embedding(self.node_count, self.embedding_dim)
        self.anonymous_embedding = nn.Embedding(self.node_count, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.embedding_dim, nhead=4, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True)
        self.semantic_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.anonymous_gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim, batch_first=True)
        self.lambda_weight = 0.5
    
    def positional_embedding(self, sequence_length):
        pe = torch.zeros(sequence_length, self.embedding_dim)
        position = torch.arange(sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embedding_dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        return pe

    def semanticwalk_encode(self, semantic_walk):
        embedded_semanticwalk = self.semantic_embedding(torch.tensor(semantic_walk, dtype=torch.long))
        embedded_semanticwalk = embedded_semanticwalk + self.positional_embedding(len(semantic_walk))
        return embedded_semanticwalk

    def anonymouswalk_encode(self, anonymous_walk):
        embedded_anonymouswalk = self.anonymous_embedding(torch.tensor(anonymous_walk, dtype=torch.long))
        # embedded_anonymouswalk = embedded_anonymouswalk + self.positional_embedding(len(anonymous_walk))
        return embedded_anonymouswalk

    def forward(self, pattern):
        embedded_semanticwalk = self.semanticwalk_encode(pattern.semantic_walk)
        embedded_anonymouswalk = self.anonymouswalk_encode(pattern.anonymous_walk)

        encoded_semanticwalk = self.semantic_transformer(embedded_semanticwalk.unsqueeze(0))
        encoded_semanticwalk = encoded_semanticwalk.mean(dim=1)
        self.encoded_semanticwalk = encoded_semanticwalk.squeeze(0)

        output, hidden = self.anonymous_gru(embedded_anonymouswalk.unsqueeze(0))
        self.encoded_anonymouswalk = hidden.squeeze()

        pattern = self.encoded_semanticwalk + self.lambda_weight*self.encoded_anonymouswalk
        return pattern
        
    