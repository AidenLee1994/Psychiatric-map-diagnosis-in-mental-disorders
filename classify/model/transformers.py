import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.linear import Linear



#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 768, num_patches: int = None, dropout: float = 0.):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n):
        x = x + self.pos_embed[:, :(n+1)]
        x = self.dropout(x)
        return x

#Norm Layer
class Norm(nn.Module):
    def __init__(self, d_model: int = 768, next_layer: nn.Module = None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.next_layer = next_layer
    def forward(self, x: torch.Tensor, **kwargs):
        x = self.norm(x)
        return self.next_layer(x, **kwargs)

#Feed Forward MLP
class FeedForward(nn.Module):
    def __init__(self, d_model: int = 768, d_mlp: int = 3072, dropout: float = 0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

#Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12, dropout: float = 0.):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

        d_head = d_model // n_head
        project_out = not (n_head == 1 and d_head == d_model)

        self.scale = d_head ** -0.5
        self.softmax = nn.Softmax(dim = -1)
        self.w_qkv = nn.Linear(d_model, d_model * 3, bias = False)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.n_head
        qkv = self.w_qkv(x).chunk(3, dim = -1)
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # Compute Attention score
        scores = torch.einsum('b h i d, b h j d -> b h i j', queries, keys) * self.scale
        attention = self.softmax(scores)

        x = torch.einsum('b h i j, b h j d -> b h i d', attention, values)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return x


class ViT(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 num_class: int = 1000, d_model: int = 768, n_head: int = 12,
                 n_layers: int = 12, d_mlp: int = 3072, channels: int = 3,
                 dropout: float = 0., pool: str = 'cls'):
        super().__init__()

        img_h, img_w = img_size, img_size
        patch_h, patch_w = patch_size, patch_size

        assert img_h % patch_h == 0, 'image dimension must be divisible by patch dimension'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.patches_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.Linear(patch_dim, d_model)
        )

        self.pos_embed = PositionalEncoding(d_model, num_patches, dropout)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool = pool

        self.transformer = Transformer(d_model, n_head, n_layers, d_mlp, dropout)
        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_class)
        )

    def forward(self, img):
        x = self.patches_embed(img)
        b, n, _ = x.shape
        class_token = repeat(self.class_token, '() n d -> b n d', b=b)
        # Concat Class Token with image patches
        x = torch.cat((class_token, x), dim=1)
        # Add Positional Encoding
        x = self.pos_embed(x, n)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # MLP Head
        x = self.mlp_head(x)
        return x


# Transformer
class Transformer(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12, n_layers: int = 12,
                 d_mlp: int = 3072, dropout: float = 0.):
        super().__init__()

        self.block = nn.ModuleList([
            Norm(d_model, MultiHeadAttention(d_model, n_head, dropout)),
            Norm(d_model, FeedForward(d_model, d_mlp, dropout))
        ])
        self.layers = nn.ModuleList([self.block for _ in range(n_layers)])

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x) + x
            x = mlp(x) + x
        return x

if __name__ == '__main__':
    a=torch.rand(16,1, 20, 20)
    vit=ViT(img_size=20,patch_size=2,num_class=3,channels=1)
    out=vit(a)
    print(out)