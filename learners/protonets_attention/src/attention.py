import torch
from torch import nn, einsum
import torch.nn.functional as F
from utils import extract_class_indices
from einops import rearrange


class CrossTransformer(nn.Module):
    def __init__(self, dim_features=512, dim_key=128, dim_value=128, temperature=1.0):
        super().__init__()
        self.scale = 1.0 / temperature
        # self.scale = torch.nn.Parameter(torch.tensor(1.0 / temperature))
        self.to_qk = nn.Conv2d(dim_features, dim_key, 1, bias = False)
        self.to_v = nn.Conv2d(dim_features, dim_value, 1, bias = False)

    def forward(self,
                support_features,  # (k1 x n1 + k2 x n2 + ... + kc x nc shuffled) x c x h x w
                query_features,    # b x c x h x w
                support_labels):   # (k1 x n1 + k2 x n2 + ... + kc x nc shuffled)
        """
        dimensions names:

        bq - batch size for query
        bs - batch size for support
        k - num classes
        n - num images in a support class (this will vary per class!)
        c - channels
        h, i - height
        w, j - width
        """

        query_q, query_v = self.to_qk(query_features), self.to_v(query_features)  # (k1 x n1 + k2 x n2 + ... + kc x nc shuffled), c, h, w
        supports_k, supports_v = self.to_qk(support_features), self.to_v(support_features) # b, c, h, w

        b, c, h, w = query_v.shape

        outs = []
        for cl in torch.unique(support_labels):
            # filter out feature vectors which have class k
            class_supports_k = torch.index_select(supports_k, 0, extract_class_indices(support_labels, cl))  # n, c, h, w
            class_supports_v = torch.index_select(supports_v, 0, extract_class_indices(support_labels, cl))  # n, c, h, w
            class_supports_k = rearrange(class_supports_k, 'n c h w -> () n c h w')  # 1, n, c, h, w
            class_supports_k = class_supports_k.expand(b, -1, -1, -1, -1)  # b, n, c, h, w
            sim = einsum('b c h w, b n c i j -> b h w n i j', query_q, class_supports_k) * self.scale
            sim = rearrange(sim, 'b h w n i j -> b h w (n i j)')
            attn = sim.softmax(dim = -1)
            attn = rearrange(attn, 'b h w (n i j) -> b h w n i j', i = h, j = w)

            class_supports_v = rearrange(class_supports_v, 'n c h w -> () n c h w')  # 1, n, c, h, w
            class_supports_v = class_supports_v.expand(b, -1, -1, -1, -1)  # b, n, c, h, w
            out = einsum('b h w n i j, b n c i j -> b c h w', attn, class_supports_v)
            out = rearrange(out, 'b c h w -> b (c h w)')
            outs.append(out)

        prototypes = torch.stack(outs)
        prototypes = rearrange(prototypes, 'k b (c h w) -> b k (c h w)', c = c, h = h, w = w)
        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')

        return prototypes, query_v, h, w


class BatchLinear(nn.Linear):
    """Helper class for linear layers on order-3 tensors.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Use a bias. Defaults to `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchLinear, self).__init__(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)
        nn.init.xavier_normal_(self.weight, gain=1)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """Forward pass through layer. First unroll batch dimension, then pass
        through dense layer, and finally reshape back to a order-3 tensor.
        Args:
              x (tensor): Inputs of shape `(batch, n, in_features)`.
        Returns:
              tensor: Outputs of shape `(batch, n, out_features)`.
        """
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.view(num_functions * num_inputs, self.in_features)
        out = super(BatchLinear, self).forward(x)
        return out.view(num_functions, num_inputs, self.out_features)


"""
    Attention modules for AttnCNP
"""


class DotProdAttention(nn.Module):
    """
    Simple dot-product attention module. Can be used multiple times for
    multi-head attention.
    """
    def __init__(self, temperature):
        super(DotProdAttention, self).__init__()
        self.temperature = temperature

    def forward(self, keys, queries, values):
        """Forward pass to implement dot-product attention. Assumes that
        everything is in batch mode.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.

        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        import pdb; pdb.set_trace()
        dk = keys.shape[-1]
        attn_logits = torch.bmm(queries, keys.permute(0, 2, 1)) / self.temperature
        attn_weights = nn.functional.softmax(attn_logits, dim=-1)
        return torch.bmm(attn_weights, values), attn_weights


class MultiHeadAttention(nn.Module):
    """Implementation of multi-head attention in a batch way. Wraps around the
    dot-product attention module.

    Args:
        embedding_dim (int): Dimensionality of embedding for keys, values,
            queries.
        num_heads (int): Number of dot-product attention heads in module.
    """

    def __init__(self,
                 embedding_dim,
                 num_heads,
                 temperature):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = self.embedding_dim // self.num_heads

        self.key_transform = BatchLinear(self.embedding_dim, self.embedding_dim,
                                         bias=False)
        self.query_transform = BatchLinear(self.embedding_dim,
                                           self.embedding_dim, bias=False)
        self.value_transform = BatchLinear(self.embedding_dim,
                                           self.embedding_dim, bias=False)
        self.attention = DotProdAttention(temperature=temperature)
        self.head_combine = BatchLinear(self.embedding_dim, self.embedding_dim)

    def forward(self, keys, queries, values):
        """Forward pass through multi-head attention module.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.

        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        # keys = self.key_transform(keys)
        # queries = self.query_transform(queries)
        # values = self.value_transform(values)

        # Reshape keys, queries, values into shape
        #     (batch_size * n_heads, num_points, head_size).
        keys = self._reshape_objects(keys)
        queries = self._reshape_objects(queries)
        values = self._reshape_objects(values)

        # Compute attention mechanism, reshape, process, and return.
        attn = self.attention(keys, queries, values)
        attn = self._concat_head_outputs(attn)
        return self.head_combine(attn)

    def _reshape_objects(self, o):
        num_functions = o.shape[0]
        o = o.view(num_functions, -1, self.num_heads, self.head_size)
        o = o.permute(2, 0, 1, 3).contiguous()
        return o.view(num_functions * self.num_heads, -1, self.head_size)

    def _concat_head_outputs(self, attn):
        num_functions = attn.shape[0] // self.num_heads
        attn = attn.view(self.num_heads, num_functions, -1, self.head_size)
        attn = attn.permute(1, 2, 0, 3).contiguous()
        return attn.view(num_functions, -1, self.num_heads * self.head_size)

