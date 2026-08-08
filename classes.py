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
        embedded_semanticwalk = self.semanticwalk_encode(pattern["semantic_walk"])
        embedded_anonymouswalk = self.anonymouswalk_encode(pattern["anonymous_walk"])

        encoded_semanticwalk = self.semantic_transformer(embedded_semanticwalk.unsqueeze(0))
        encoded_semanticwalk = encoded_semanticwalk.mean(dim=1)
        self.encoded_semanticwalk = encoded_semanticwalk.squeeze(0)

        output, hidden = self.anonymous_gru(embedded_anonymouswalk.unsqueeze(0))
        self.encoded_anonymouswalk = hidden.squeeze()

        pattern = self.encoded_semanticwalk + self.lambda_weight*self.encoded_anonymouswalk
        return pattern

class PatternAggregator(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # The Transformer Encoder layer to process the sequence of pattern embeddings.
        # batch_first=True makes it expect input of shape (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation="relu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, pattern_embeddings):
        # Input shape: (num_patterns, embedding_dim)
        
        # 1. Unsqueeze: Add a batch dimension.
        # nn.TransformerEncoder with batch_first=True expects (batch_size, sequence_length, embedding_dim).
        # We are processing a single graph's patterns, so batch_size=1 and sequence_length=num_patterns.
        # Shape becomes: (1, num_patterns, embedding_dim)
        x = pattern_embeddings.unsqueeze(0)
        
        # 2. Transformer: Apply self-attention across patterns.
        # This allows each local pattern to attend to every other pattern in the graph, 
        # learning their global structural relationships.
        # Shape remains: (1, num_patterns, embedding_dim)
        x = self.transformer(x)
        
        # 3. Pooling: Aggregate the enriched pattern embeddings into ONE graph embedding.
        # We use mean pooling over the sequence dimension (dim=1) as per standard 
        # set-aggregation strategies to form a fixed-size representation.
        # Shape becomes: (1, embedding_dim)
        graph_embedding = x.mean(dim=1)
        
        # 4. Squeeze: Remove the batch dimension.
        # This returns the final 1D graph embedding tensor.
        # Shape becomes: (embedding_dim,)
        graph_embedding = graph_embedding.squeeze(0)
        
        return graph_embedding