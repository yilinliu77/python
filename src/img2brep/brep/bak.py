
def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def first(it):
    return it[0]


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def is_empty(l):
    return len(l) == 0


def is_tensor_empty(t: Tensor):
    return t.numel() == 0


def set_module_requires_grad_(
        module: Module,
        requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad



def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim=dim)


def pad_at_dim(t, padding, dim=-1, value=0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value=value)


def pad_to_length(t, length, dim=-1, value=0, right=True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim=dim, value=value)



def discretize(
        t: Tensor,
        continuous_range: Tuple[float, float],
        num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min=0, max=num_discrete - 1)


def undiscretize(
        t: Tensor,
        continuous_range=Tuple[float, float],
        num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()
    t += 0.5
    t /= num_discrete

    return t * (hi - lo) + lo


def gaussian_blur_1d(
        t: Tensor,
        sigma: float = 1.
) -> Tensor:
    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype=dtype, device=device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c=channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding=half_width, groups=channels)

    return rearrange(out, 'b c n -> b n c')


# resnet block

class PixelNorm(Module):
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim=dim, eps=self.eps) * sqrt(x.shape[dim])


class SqueezeExcite(Module):
    def __init__(
            self,
            dim,
            reduction_factor=4,
            min_dim=16
    ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min=1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)


class Block(Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            dropout=0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = PixelNorm(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class ResnetBlock(Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            *,
            dropout=0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
            self,
            x,
            mask=None
    ):
        res = self.residual_conv(x)
        h = self.block1(x, mask=mask)
        h = self.block2(h, mask=mask)
        h = self.excite(h, mask=mask)
        return h + res

class Decoder(nn.Module):
    def __init__(self,
                 decoder_dims_through_depth,
                 init_decoder_conv_kernel,
                 init_decoder_dim,
                 dim_codebook_edge,
                 dim_codebook_face,
                 resnet_dropout
                 ):
        super(Decoder, self).__init__()
        # For edges
        self.edge_decoder_init = nn.Sequential(
            nn.Linear(dim_codebook_edge, init_decoder_dim),
            nn.SiLU(),
            nn.LayerNorm(init_decoder_dim),
        )

        self.edge_decoder = ModuleList([])
        curr_dim = init_decoder_dim
        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout=resnet_dropout)
            self.edge_decoder.append(resnet_block)
            curr_dim = dim_layer

        self.to_edge = nn.Sequential(
            nn.Linear(curr_dim, 20 * 3),
            Rearrange('... (v c) -> ... v c', v=20)
        )

        # For faces
        self.face_decoder_init = nn.Sequential(
            nn.Conv1d(dim_codebook_face, init_decoder_dim,
                      kernel_size=init_decoder_conv_kernel, padding=init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.face_decoder = ModuleList([])

        curr_dim = init_decoder_dim
        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout=resnet_dropout)

            self.face_decoder.append(resnet_block)
            curr_dim = dim_layer

        self.to_face = nn.Sequential(
            nn.Linear(curr_dim, 20 * 20 * 3),
            Rearrange('... (v w c) -> ... v w c', v=20, w=20)
        )

    def forward(self, v_edge_embeddings, v_face_embeddings):
        x = self.edge_decoder_init(v_edge_embeddings[:, :, None])
        for resnet_block in self.edge_decoder:
            x = resnet_block(x)
        recon_edges = self.to_edge(x[..., 0])

        x = self.face_decoder_init(v_face_embeddings[:, :, None])
        for resnet_block in self.face_decoder:
            x = resnet_block(x)
        recon_faces = self.to_face(x[..., 0])
        return recon_edges, recon_faces

# self.decoder = Decoder(
#     decoder_dims_through_depth=(
#         128, 128, 128, 128,
#         192, 192, 192, 192,
#         256, 256, 256, 256, 256, 256,
#         384, 384, 384
#     ),
#     init_decoder_conv_kernel=7,
#     init_decoder_dim=256,
#     dim_codebook_edge=dim_codebook_edge,
#     dim_codebook_face=dim_codebook_face,
#     resnet_dropout=0,
# )