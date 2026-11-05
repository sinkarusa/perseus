"""LFM2 (LiquidAI LFM2 / LFM2.5, transformers ``models/lfm2``) decode on ``perseus.impl``, built on the Llama port
(examples/llama_from_primitives: cachemir_complex packing, GQA sub-runtime, RMSNorm seed + Newton, THOR SiLU /
softmax, the BOS plaintext prefix, the plan machinery).

What LFM2 adds (layer schedule from config.layer_types):
  ShortConv   conv layers: in_proj -> (B + iC) fused complex linear and x linear, h = B x (ct x ct), the depthwise
              causal conv over (h_{t-2}, h_{t-1}, h_t) as three plaintext-mask products (the per-channel kernel
              taps, the lane-0 selection and the Re/Im 1/2 folded into the masks), y = C out (ct x ct),
              out_proj. Decode state: the two last encrypted h per conv layer (conv.ConvState).
  QK-norm     attention layers: RMSNorm over every 64-slot head of q (32 heads) and k (8 KV heads) BEFORE RoPE,
              on the refreshed K / V / Q payload of the KV_LANES / QKV_LANES push (qknorm.qk_norm): one square,
              one tH ladder, one Chebyshev seed + Newton inverse sqrt for all heads and both q and k; gamma and
              the output scale ride the numerator mask.
  The final RMSNorm (`embedding_norm`) and the tied LM head run on the client (model.decode_token(head=False)).

Modules: config (configs.json -> structs), weights (bf16 safetensors -> slot vectors), conv, qknorm, block, model,
prefix (plaintext BOS / prompt state), env (rotation keys), calibrate (fits from collected site statistics).
"""
