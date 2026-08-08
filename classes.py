import torch
import torch.nn as nn
class Pattern():
    def __init__(self, non_anon, anon):
        self.semantic_walk=non_anon
        self.anonymous_walk=anon

class PatternEncoder(nn.Module):
    def __init__(self, node_count, embedding_dim, nhead=4, encoder_layers=2, dim_feedforward=2048):
        super().__init__()
        self.node_count = node_count
        self.embedding_dim = embedding_dim
        self.semantic_embedding = nn.Embedding(self.node_count, self.embedding_dim)
        self.anonymous_embedding = nn.Embedding(self.node_count, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation="relu", batch_first=True)
        self.semantic_transformer = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.anonymous_gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim, batch_first=True)
        self.lambda_weight = 0.5
    
    def positional_embedding(self, sequence_length):
        device = self.semantic_embedding.weight.device
        pe = torch.zeros(sequence_length, self.embedding_dim, device=device)
        position = torch.arange(sequence_length, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / self.embedding_dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        return pe

    def semanticwalk_encode(self, semantic_walk):
        device = self.semantic_embedding.weight.device
        embedded_semanticwalk = self.semantic_embedding(torch.tensor(semantic_walk, dtype=torch.long, device=device))
        embedded_semanticwalk = embedded_semanticwalk + self.positional_embedding(len(semantic_walk))
        return embedded_semanticwalk

    def anonymouswalk_encode(self, anonymous_walk):
        device = self.anonymous_embedding.weight.device
        embedded_anonymouswalk = self.anonymous_embedding(torch.tensor(anonymous_walk, dtype=torch.long, device=device))
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

class GraphClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # A lightweight MLP for downstream prediction.
        # The representation power lies in the GPM encoders; the MLP just projects 
        # the rich structural embedding into the classification space.
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout for regularization
            nn.Linear(self.embedding_dim // 2, self.num_classes)
        )

    def forward(self, graph_embedding):
        # Input shape expected from PatternAggregator: (embedding_dim,)
        
        # Linear layers expect a batch dimension. If the input is unbatched (1D),
        # we add a temporary batch dimension.
        is_unbatched = graph_embedding.dim() == 1
        if is_unbatched:
            # Shape becomes: (1, embedding_dim)
            graph_embedding = graph_embedding.unsqueeze(0)
            
        # Pass through the MLP.
        # Shape becomes: (batch_size, num_classes)
        logits = self.mlp(graph_embedding)
        
        # If we artificially added a batch dimension, remove it before returning.
        if is_unbatched:
            # Shape becomes: (num_classes,)
            logits = logits.squeeze(0)
            
        return logits

class GPMModel(nn.Module):
    def __init__(self, node_count, embedding_dim, num_classes,
                 nhead=4, encoder_layers=2, encoder_feedforward=2048,
                 aggregator_heads=4, aggregator_layers=2, aggregator_feedforward=2048):
        super().__init__()
        self.encoder = PatternEncoder(node_count, embedding_dim,
                                      nhead=nhead,
                                      encoder_layers=encoder_layers,
                                      dim_feedforward=encoder_feedforward)
        self.aggregator = PatternAggregator(embedding_dim,
                                            num_heads=aggregator_heads,
                                            num_layers=aggregator_layers,
                                            dim_feedforward=aggregator_feedforward)
        self.classifier = GraphClassifier(embedding_dim, num_classes)

    def forward(self, patterns):
        """
        Forward pass for a single graph.
        
        Args:
            patterns: A list of pattern dictionaries (each with 'semantic_walk' and 'anonymous_walk').
                      These represent the sampled walks for ONE graph.
        
        Returns:
            logits: Prediction logits of shape (num_classes,)
        """
        # 1. Encode all patterns using the PatternEncoder.
        encoded_patterns = []
        for p in patterns:
            encoded_patterns.append(self.encoder(p))
            
        # 2. Stack into a tensor of shape (num_patterns, embedding_dim)
        pattern_tensor = torch.stack(encoded_patterns)
        
        # 3. Aggregate into a single graph embedding of shape (embedding_dim,)
        graph_embedding = self.aggregator(pattern_tensor)
        
        # 4. Classify the graph embedding to get logits of shape (num_classes,)
        logits = self.classifier(graph_embedding)
        
        return logits