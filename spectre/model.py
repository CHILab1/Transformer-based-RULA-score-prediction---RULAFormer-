"""Original multi-task SimpleViT architecture; validated dimensions and registered positions."""
import torch
from torch import nn

class Patchify(nn.Module):
    """Same patch ordering as einops b c (h p1) (w p2) -> b (h w) (p1 p2 c)."""
    def __init__(self, p1, p2):
        super().__init__()
        self.p1, self.p2 = p1, p2

    def forward(self, x):
        b, c, height, width = x.shape
        h, w = height // self.p1, width // self.p2
        return x.reshape(b, c, h, self.p1, w, self.p2).permute(0, 2, 4, 3, 5, 1).reshape(b, h*w, -1)

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if dim < 8 or dim % 4:
        raise ValueError("Embedding dimension must be >= 8 and divisible by 4")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = (t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, output_class_dims=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        if any(v <= 0 for v in [image_height, image_width, patch_height, patch_width, dim, depth, heads, mlp_dim, channels, dim_head]):
            raise ValueError("Model dimensions must be positive")
        if image_height % patch_height or image_width % patch_width:
            raise ValueError("Image dimensions must be divisible by patch size")
        if output_class_dims is None or len(output_class_dims) != num_classes or any(c < 2 for c in output_class_dims):
            raise ValueError("One output dimension >= 2 is required per task")
        self.image_shape = (channels, image_height, image_width)

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Patchify(patch_height, patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.register_buffer("pos_embedding", posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.attention_heads = nn.ModuleList([
        nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
            nn.Sigmoid()
        ) for _ in range(num_classes)
    ])

        self.pool = "mean"
        self.to_latent = nn.Identity()
                                
        self.linear_head = nn.ModuleList()
        
        self.output_class_dims= output_class_dims
        
        for count in output_class_dims:
            self.linear_head.append(nn.Linear(dim, count)) 

    def forward(self, img):
        if img.ndim != 4 or tuple(img.shape[1:]) != self.image_shape:
            raise ValueError(f"Expected [batch, {self.image_shape}], got {tuple(img.shape)}")
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        
        output= [head(x* att(x)) for head, att in zip(self.linear_head, self.attention_heads)] 
        
        return output
