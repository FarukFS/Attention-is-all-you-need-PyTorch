import torch
import torchvision
import torch.nn as nn


class SelfAttention(nn.Module):
    # We are going to split the embedding into different parts/heads.
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert (self.heads_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # For each input, each attention head produces a different output vector. Here, we concatenate these.
        self.fc_out = nn.Linear(heads*self.heads_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get the number of training examples
        n = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces.
        values = values.reshape(n, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(n, key_len, self.heads, self.heads_dim)
        queries = query.reshape(n, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # query shape = (N, query_len, heads, heads_dim)
        # keys shape = (N, key_len, heads, heads_dim)
        # We want weight shape = (N, heads, query_len, key_len)
        weight = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        if mask is not None:
            # If the element of the mask is 0, we shut 'off'. This is because for y1, we only want to look at x1, for y2 we only look at
            # x1 and x2, for y3 we look at x1, x2,x3 and so... If we don't do that, then we are essentially learning a mapping from
            # input to output.
            weight = weight.masked_fill(mask == 0, float("-1e20"))

        # weight is of shape (N, heads, query_len, key_len), so we normalize across key_len.
        # Suppose query_len is target len and key_len is input len. This will assign weights (sum to 1) to each input word according to a
        # specific output word.
        attention = torch.softmax(weight / (self.embed_size ** (1/2)), dim=3)

        # attention shape = (N, heads, query_len, key_len)
        # values shape = (N, value_len, heads, heads_dim)
        # desired out shape = (N, query_len, heads, heads_dim). Key_len and Value_len are always equal, so let's call them 'l'.
        # After the operation, we concatenate all the outputs.
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(n, query_len, self.heads*self.heads_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        # BatchNorm takes the average across the batch, LayerNorm takes an average for every single example.
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # Why query?
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_size, n_layers, heads, device, forward_expansion, dropout, max_length):
        # max_length is the max sentence length (used for positional embedding).
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, seq_length = x.shape
        # Create an array (0,1,...,seq_length) for every example n.
        positions = torch.arange(0, seq_length).expand(n, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # At this stage, value, key and query will be the same.
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    # Whenever the sentence length is less than the required, we need to pad. We can use source_mask to avoid computing unnecessarily. Opt.
    # The target mask is essential because as mentioned before, we don't want our model to be able to look forward into the sequence.
    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        # These were computed before with the encoder.
        out = self.transformer_block(value, key, query, source_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                                     for _ in range(num_layers)])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, source_mask, target_mask):
        n, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(n, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            # We calculated key and value from the encoder (input sentence). We only need query from decoder (output sentence).
            x = layer(x, encoder_out, encoder_out, source_mask, target_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            source_vocab_size,
            target_vocab_size,
            source_pad_idx,
            target_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device=torch.device('cuda') if torch.cuda.is_available() else 'cpu',
            max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(source_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_source_mask(self, source):
        # We make the shape (n, 1, 1, source_length) just for it to work with the network.
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        n, target_length = target.shape
        target_mask = torch.tril(torch.ones(target_length, target_length)).expand(n, 1, target_length, target_length)
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoder_source = self.encoder(source, source_mask)
        out = self.decoder(target, encoder_source, source_mask, target_mask)
        return out


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    source_pad_index = 0
    target_pad_index = 0
    source_vocab_size = 10
    target_vocab_size = 10
    model = Transformer(source_vocab_size, target_vocab_size, source_pad_index, target_pad_index, device=device).to(device)

    # Include all examples except the last one. We want it to predict the next sentence.
    out = model(x, target[:, :-1])
    print(out)
    print(out.shape)

