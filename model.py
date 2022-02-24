import torch.nn as nn

"""
Majority of code is shamelessly copied from here:

https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial

And then severely hacked together to make more sense,
be more readable, better documented and expanded upon
"""

class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        """
        The numbers below are from BERT; larger ones are possible
        but in the case of Token embeddings won't make sense.

        This is the embedding layer using:
            - a Token(Word) Embedding
            - a Positional Embedding (max sentence size)
            - a Token type Embedding (2 types)
            - a Layer that applies normalisation

        See:
            https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        """
        ddim   = 768
        maxlen = 512
        n_seg  = 2
        self.tok_embed = nn.Embedding(vocab_size, ddim)  	# token embedding
        self.pos_embed = nn.Embedding(maxlen, ddim)  		# position embedding
        self.seg_embed = nn.Embedding(n_segments, ddim)  	# segment(token type) embedding
        self.norm = nn.LayerNorm(ddim)


    def forward(self, x, seg):
        """Forward Prop:
            x:   is the tensor of encoded input ids
            seg: is the tensor of segment ids

        See:
            https://pytorch.org/docs/stable/generated/torch.arange.html
            https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

        >>>The gist of this is:
            - get the sequence length `x.size(1)`
            - greate a range from 0 up to `seq_len` which is of Long type
            - `unsqueeze` will add a dimension at dim = 0, and then expand it as `x`
            - finally, we propagate `x` through token embedding, `pos` through pos embedding
              and `seg` through segment embedding, eventually adding them all together
            - we also normalise the output before returning it

        The above operation is at the core of BERT's embedding process.

        https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  			# (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


def get_attn_pad_mask(seq_q, seq_k):
    """
    Create an Attention Mask which is padded.

    The documentation says that the attention mask is an optional argument used when batching sequences together.
    This argument indicates to the model which tokens should be attended to, and which should not.
    For example, the tokenizer encoding methods return this attention mask,
    a binary tensor indicating the position of the padded indices so that the model does not attend to them,
    which makes sense.

    See here for more:
        https://huggingface.co/docs/transformers/glossary#attention-mask
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class EncoderLayer(nn.Module):
    """
    Encoder Layer for BERT reconstructed for brevity
    """
    # TODO: read, document, understand, explain
    def __init__(self):
        """
        There's two parts to the Encoder Layer:
            - Multi Head Attention
            - Powsise Feed Forward

        The original code implemented MultiHead Attention, but PyTorch has it's own implementation:
            https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

        It basically takes the embedding dimentions of the model,
        and the number of parallel attention heads. embed_dim will be split across num_heads,
        so that embed_dim // num_heads

        When used, it expects a query, key and value. Optionally it will accept key padding mask and attention mask.
        For BERT, we will be using an Attention mask.

        For more info on the Heads, read:
            https://stackoverflow.com/questions/69436845/bert-heads-count

        TODO: PosWiseFeedForwardNet
        """
        embed_dim = 768
        num_heads = 12
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn



class BERT(nn.Module):
   def __init__(self):
       super(BERT, self).__init__()
       self.embedding = Embedding()
       self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
       self.fc = nn.Linear(d_model, d_model)
       self.activ1 = nn.Tanh()
       self.linear = nn.Linear(d_model, d_model)
       self.activ2 = gelu
       self.norm = nn.LayerNorm(d_model)
       self.classifier = nn.Linear(d_model, 2)
       # decoder is shared with embedding layer
       embed_weight = self.embedding.tok_embed.weight
       n_vocab, n_dim = embed_weight.size()
       self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
       self.decoder.weight = embed_weight
       self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

   def forward(self, input_ids, segment_ids, masked_pos):
       output = self.embedding(input_ids, segment_ids)
       enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
       for layer in self.layers:
           output, enc_self_attn = layer(output, enc_self_attn_mask)
       # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
       # it will be decided by first token(CLS)
       h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
       logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

       masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]

       # get masked position from final output of transformer.
       h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
       h_masked = self.norm(self.activ2(self.linear(h_masked)))
       logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

       return logits_lm, logits_clsf
