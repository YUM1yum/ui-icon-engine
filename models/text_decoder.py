# Stage 3: Transformer Decoder (KV Cache 포함)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    KV Caching을 지원하는 Causal Self Attention
    """
    def __init__(self, embed_dim, num_heads, max_len=128):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V Projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Causal Mask (Register buffer to save it in state_dict)
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                    .view(1, 1, max_len, max_len))

    def forward(self, x, past_kv=None):
        B, T, C = x.size() # Batch, Time(Seq Len), Channels
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # --- KV Caching Logic ---
        if past_kv is not None:
            past_k, past_v = past_kv
            # 현재 스텝의 k, v를 과거의 것과 연결
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # Update KV cache for next step
        current_kv = (k, v)

        # Attention Score 계산 (Flash Attention 지원 API 사용)
        # Inference 시에는 마지막 토큰에 대해서만 Query를 던지므로 마스킹 처리가 조금 다름
        if past_kv is not None:
            # Inference: Query는 1개(현재 토큰), Key/Val은 전체(과거+현재)
            # 마스크 불필요 (이미 지난 정보는 다 볼 수 있음)
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Training: Standard Causal Masking
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output), current_kv

class CrossAttention(nn.Module):
    """
    Decoder(Text)가 Encoder(Visual Feature)를 참조하는 부분
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim) # From Vision
        self.v_proj = nn.Linear(embed_dim, embed_dim) # From Vision
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, visual_context):
        # x: [B, T, C] (Text)
        # visual_context: [B, 1, C] (From Fusion Bridge)
        
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Visual Context는 시퀀스 길이 1(요약된 벡터)이라고 가정 (Broadcasting)
        # 만약 Visual Token이 여러개라면 T_v 차원이 됨.
        k = self.k_proj(visual_context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual_context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = CausalSelfAttention(embed_dim, num_heads)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x, visual_context, past_kv=None):
        # 1. Self Attention with Residual
        attn_out, new_kv = self.self_attn(self.ln1(x), past_kv)
        x = x + attn_out
        
        # 2. Cross Attention with Residual
        # Visual Context는 고정되어 있으므로 캐싱 불필요 (혹은 외부에서 고정)
        x = x + self.cross_attn(self.ln2(x), visual_context)
        
        # 3. Feed Forward with Residual
        x = x + self.mlp(self.ln3(x))
        
        return x, new_kv

class UITextDecoder(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(128, embed_dim) # Max Len 128
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying (Optional but recommended for size reduction)
        self.token_embedding.weight = self.head.weight

    def forward(self, input_ids, visual_context, past_key_values=None):
        """
        input_ids: [B, T]
        visual_context: [B, 1, embed_dim] (From Bridge)
        past_key_values: List of (k, v) tuples for each layer
        """
        B, T = input_ids.size()
        
        # Inference 시 (past_key_values 존재)에는 마지막 토큰의 포지션만 계산
        if past_key_values is not None:
            # past_key_values[0][0] shape: [B, num_heads, past_T, head_dim]
            past_length = past_key_values[0][0].size(2)
            pos_ids = torch.arange(past_length, past_length + T, device=input_ids.device)
        else:
            pos_ids = torch.arange(0, T, device=input_ids.device)
            
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        
        new_key_values = []
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, layer_kv = layer(x, visual_context, past_kv)
            new_key_values.append(layer_kv)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, new_key_values