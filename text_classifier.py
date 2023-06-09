import torch
from torch import nn
from einops import rearrange, repeat

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class NoSoftMaxAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2))

        #attn = self.attend(dots)

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class QuinticAttetion(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 5, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(5, dim = -1)
        q1, k1, q2, k2, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2))
        dots2 = torch.matmul(q2, k2.transpose(-1, -2))
        dots = dots1@dots2
        #attn = self.attend(dots)

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attention_type, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        if attention_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'no_softmax':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, NoSoftMaxAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'quintic':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QuinticAttetion(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, weight_matrix, train = False):
        super().__init__()
        vocab_size, embedding_dim = weight_matrix.size()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.load_state_dict({'weight':weight_matrix})
        if not train:
            self.embedding_layer.requires_grad = False
    def forward(self, x):
        return self.embedding_layer(x)

class Review_Classifier(nn.Module):
    def __init__(self, *, embedding_weights, sentence_length, embedding_dim, num_classes, dim, depth, heads, mlp_dim, attention_type = 'standard',  pool = 'cls',  dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        

        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        
        self.pos_embedding = nn.Parameter(torch.randn(1, sentence_length + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = dropout, attention_type = attention_type)

        self.pool = pool
        #self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        if dim == embedding_dim:
            self.to_model_dim = nn.Linear(embedding_dim, dim)
        else:
            self.to_model_dim = nn.Identity(dim)

        self.embedding = EmbeddingLayer(embedding_weights)

    def forward(self, x):
        x = self.embedding(x)
        #print(x.shape)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.to_model_dim(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)
