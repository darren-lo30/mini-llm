# Following tutorial: https://www.youtube.com/watch?v=zy8ChVd_oTM&t=25575s
import torch

import triton
import triton.language as tl

@triton.jit
def _attention_forward_helper(
  O_block,
  maxs,
  sums,
  Q_block,
  K_block_ptr,
  V_block_ptr,
  seq_idx,
  softmax_scale,
  BLOCK_SIZE_Q: tl.constexpr,
  BLOCK_SIZE_KV: tl.constexpr,
  STAGE: tl.constexpr,
  offsets_q: tl.constexpr,
  offsets_kv: tl.constexpr,
  SEQ_LEN: tl.constexpr,
):
  
  if STAGE == 1: # Before diagonal
    lo, hi = 0, seq_idx * BLOCK_SIZE_Q
  elif STAGE == 2: # On diagonal
    lo, hi = seq_idx * BLOCK_SIZE_Q, (seq_idx + 1) * BLOCK_SIZE_Q
    lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
  else: # STAGE == 3, Non causal attention
    lo, hi = 0, SEQ_LEN

  K_block_ptr = tl.advance(K_block_ptr, (0, lo))
  V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

  for start_kv in range(lo, hi, BLOCK_SIZE_KV):
    start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
    K_block = tl.load(K_block_ptr)
    # Compute QK^T
    QK_block = tl.dot(Q_block, K_block) * softmax_scale
    if STAGE == 2:
      mask = offsets_q[:, None] >= (start_kv + offsets_kv[None, :])
      QK_block = QK_block + tl.where(mask, 0, -1.0e6)
      new_row_maxs = tl.maximum(maxs, tl.max(QK_block, 1))
      QK_block -= new_row_maxs[:, None]
    else:
      new_row_maxs = tl.maximum(maxs, tl.max(QK_block, 1))
      QK_block = QK_block - new_row_maxs[:, None]

    P_block = tl.math.exp(QK_block) 
    row_sums = tl.sum(P_block, 1)
    correction_factor = tl.math.exp(maxs - new_row_maxs)

    sums = sums * correction_factor + row_sums

    V_block = tl.load(V_block_ptr)

    P_block = P_block.to(tl.float16)

    # Correct previous rows if a new max exists
    O_block = O_block * correction_factor[:, None]
    # O = O + PV
    O_block = tl.dot(P_block, V_block, O_block)

    maxs = new_row_maxs

    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))

  return O_block, maxs, sums

@triton.autotune(
  [
    triton.Config(
        {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_SIZE_Q in [64, 128]
    for BLOCK_SIZE_KV in [32, 64]
    for num_stages in ([3, 4, 7])
    for num_warps in [2, 4]
  ],
  key=["SEQ_LEN", "EMB_DIM"],
)
@triton.jit
@triton.jit
def _attention_forward(
  Q,
  K,
  V,
  softmax_scale,
  M, # Save maximum and log sum for each query
  O,
  stride_Q_batch,
  stride_Q_head,
  stride_Q_seq,
  stride_Q_dim,
  stride_K_batch,
  stride_K_head,
  stride_K_seq,
  stride_K_dim,
  stride_V_batch,
  stride_V_head,
  stride_V_seq,
  stride_V_dim,
  stride_O_batch,
  stride_O_head,
  stride_O_seq,
  stride_O_dim,
  BATCH_SIZE: tl.constexpr,
  NUM_HEADS: tl.constexpr,
  SEQ_LEN: tl.constexpr,
  EMB_DIM: tl.constexpr,
  BLOCK_SIZE_Q: tl.constexpr,
  BLOCK_SIZE_KV: tl.constexpr,
  STAGE: tl.constexpr,
):
  seq_idx = tl.program_id(0)

  batch_and_head_idx = tl.program_id(1)

  batch_idx = batch_and_head_idx // NUM_HEADS
  head_idx = batch_and_head_idx % NUM_HEADS

  # Gets the offset inside Q, K, V since they are flattened to 1D arrays
  qkv_offset = batch_idx.to(tl.int64) * stride_Q_batch + head_idx.to(tl.int64) * stride_Q_head

  # Defines the block in Q we are working with
  # Q[batch_idx, head_idx, seq_idx * BLOCK_SIZE_Q:  seq_idx * BLOCK_SIZE_Q + BLOCK_SIZE_Q, :]
  Q_block_ptr = tl.make_block_ptr(
    base = Q + qkv_offset, # Gets correct batch and head offset
    shape = (SEQ_LEN, EMB_DIM),
    strides = (stride_Q_seq, stride_Q_dim),
    offsets = (seq_idx * BLOCK_SIZE_Q, 0),
    block_shape = (BLOCK_SIZE_Q, EMB_DIM),
    order = (1, 0)
  )
  
  V_block_ptr = tl.make_block_ptr(
    base = V + qkv_offset,
    shape = (SEQ_LEN, EMB_DIM),
    strides = (stride_V_seq, stride_V_dim),
    offsets = (0, 0),
    block_shape = (BLOCK_SIZE_KV, EMB_DIM),
    order = (1, 0)
  )

  # Transpose K
  K_block_ptr = tl.make_block_ptr(
    base = K + qkv_offset,
    shape = (EMB_DIM, SEQ_LEN),
    strides = (stride_K_dim, stride_K_seq),
    offsets = (0, 0),
    block_shape = (EMB_DIM, BLOCK_SIZE_KV),
    order = (0, 1)
  )

  O_block_ptr = tl.make_block_ptr(
    base = O + qkv_offset, # Gets correct batch and head offset
    shape = (SEQ_LEN, EMB_DIM),
    strides = (stride_Q_seq, stride_Q_dim),
    offsets = (seq_idx * BLOCK_SIZE_Q, 0),
    block_shape = (BLOCK_SIZE_Q, EMB_DIM),
    order = (1, 0)
  )
  
  offsets_q = seq_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
  offsets_kv = tl.arange(0, BLOCK_SIZE_KV)

  # For each element in the sequence, 
  maxs = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
  sums = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

  O_block = tl.zeros([BLOCK_SIZE_Q, EMB_DIM], dtype=tl.float32)

  Q_block = tl.load(Q_block_ptr)

  # Causal attention, mask next tokens
  if STAGE == 1 or STAGE == 3:
    O_block, maxs, sums = _attention_forward_helper(
      O_block,
      maxs,
      sums,
      Q_block,
      K_block_ptr,
      V_block_ptr,
      seq_idx,
      softmax_scale,
      BLOCK_SIZE_Q,
      BLOCK_SIZE_KV,
      4 - STAGE,
      offsets_q,
      offsets_kv,
      SEQ_LEN
    )
  if STAGE == 3:
    O_block, maxs, sums = _attention_forward_helper(
      O_block,
      maxs,
      sums,
      Q_block,
      K_block_ptr,
      V_block_ptr,
      seq_idx,
      softmax_scale,
      BLOCK_SIZE_Q,
      BLOCK_SIZE_KV,
      2,
      offsets_q,
      offsets_kv,
      SEQ_LEN
    )

  # Used in backprop to recompute softmax
  maxs += tl.math.log(sums)
  O_block /= sums[:, None]
  # Find position of sequences being worked on
  m_ptrs = M + batch_and_head_idx * SEQ_LEN + offsets_q
  tl.store(m_ptrs, maxs)
  tl.store(O_block_ptr, O_block.to(O.type.element_ty))

@triton.jit
def _attention_backward_dk_dv(
  Q, 
  K,
  V,
  softmax_scale,
  dO,
  dQ,
  dK,
  dV,
  M,
  D,
  stride_batch,
  stride_head,
  stride_seq,
  stride_dim,
  NUM_HEADS,
  SEQ_LEN,
  BLOCK_Q: tl.constexpr,
  BLOCK_KV: tl.constexpr,
  EMB_DIM: tl.constexpr,
  STAGE: tl.constexpr
):
  batch_and_head_idx = tl.program_id(2)
  batch_idx = batch_and_head_idx // NUM_HEADS
  head_idx = batch_and_head_idx % NUM_HEADS

  offset_batch_head = (stride_batch * batch_idx + stride_head * head_idx).to(tl.int64)

  # No embedding dimension for M and D
  offset_batch_head_seq = (batch_and_head_idx * SEQ_LEN).to(tl.int64)

  # Get into right batch and head
  # (Batch, head, seq, emb_dim)
  Q += offset_batch_head
  K += offset_batch_head
  V += offset_batch_head
  dO += offset_batch_head
  dQ += offset_batch_head
  dK += offset_batch_head
  dV += offset_batch_head

  # (Batch, head, seq)
  M += offset_batch_head_seq
  D += offset_batch_head_seq

  offs_dim = tl.arange(0, EMB_DIM)

  # Get into right KV block
  index_block_kv = tl.program_id(0)
  start_kv = index_block_kv * BLOCK_KV

  offs_kv = start_kv + tl.arange(0, BLOCK_KV)

  dV_block = tl.zeros([BLOCK_KV, EMB_DIM], dtype=tl.float32)
  dK_block = tl.zeros([BLOCK_KV, EMB_DIM], dtype=tl.float32)

  # Load K and V block to SRAM
  K_block = tl.load(
    K + offs_kv[:, None] *  stride_seq + offs_dim[None, :] * stride_dim
  )
  V_block = tl.load(
    V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  )
  
  # Offsets for query 
  offs_q = tl.arange(0, BLOCK_Q)

  # Transpose trick
  # If BLOCK_Q = 3, head_dim = 4
  # Block is 3 x 4
  # We get
  # 0 1 2 + col(0, 1*4, 2*4)
  # 
  # 0 4 8
  # 1 5 9
  # 2 6 10
  # 3 7 11
  # Reformat as a 2D array
  qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
  dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

  curr_q = 0
  # Iterate over all queries
  num_steps = SEQ_LEN // BLOCK_Q
  for blk_idx in range(num_steps):
    qT_block = tl.load(qT_ptrs)
    offs_q = curr_q + tl.arange(0, BLOCK_Q)
    m = tl.load(M + offs_q)
  
    QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
    # e(x_i - (max(x_i) + log(sum e^{x_i}))
    # e(x_i - max(x_i)) / sum e^{x_i} <- softmax
    P_T_block = tl.math.exp(QK_T_block - m[None, :])

    # Causal mask, ignore queries after current kv
    if STAGE == 3:
      mask_block = (
        offs_q[None, :] >= offs_kv[:, None]
      )
      P_T_block = tl.where(mask_block, P_T_block, 0.0)

    dO_block = tl.load(dO_ptrs)
    dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

    Di = tl.load(D + offs_q)
    dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
    dS_T_block = P_T_block * (dpT_block - Di[None, :])
    dS_T_block = dS_T_block.to(tl.float16)
    dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
    curr_q += BLOCK_Q
    # Move to next Q block
    qT_ptrs += BLOCK_Q * stride_seq
    dO_ptrs += BLOCK_Q * stride_seq

  dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  tl.store(dV_block_ptrs, dV_block)

  dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  tl.store(dK_block_ptrs, dK_block)

@triton.jit
def _attention_backward_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    EMB_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, EMB_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Load Q and O
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, EMB_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)
    
    # Rearrange as 2d array, Transpose both
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    # Iterate through K, V blocks
    for blk_idx in range(num_steps):
      K_T_block = tl.load(kT_ptrs)
      V_T_block = tl.load(vT_ptrs)
      QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
      P_block = tl.math.exp(QK_block - M_block)

      if STAGE == 3:
        offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
        mask_block = offs_q[:, None] >= offs_kv[None, :]
        P_block = tl.where(mask_block, P_block, 0.0)

      dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
      dS_block = P_block * (dP_block - Di[:, None])
      dS_block = dS_block.to(tl.float16)
      dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
      curr_kv += BLOCK_KV
      kT_ptrs += BLOCK_KV * stride_seq
      vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)

@triton.jit
def _attention_backward_di(
  O,
  dO,
  D,
  SEQ_LEN: tl.constexpr,
  BLOCK_SIZE_Q: tl.constexpr,
  EMB_DIM: tl.constexpr,
):
  seq_idx = tl.program_id(0)
  batch_and_head_idx = tl.program_id(1)
  
  offsets_q = seq_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
  offsets_dim = tl.arange(0, EMB_DIM)

  O_block = tl.load(
    O + batch_and_head_idx * SEQ_LEN * EMB_DIM + offsets_q[:, None] * EMB_DIM + offsets_dim[None, :]
  )

  dO_block = tl.load(
    dO + batch_and_head_idx * SEQ_LEN * EMB_DIM + offsets_q[:, None] * EMB_DIM + offsets_dim[None, :]
  ).to(tl.float32)

  D_block = tl.sum(dO_block * O_block, axis=1)

  D_block_ptrs = D + batch_and_head_idx * SEQ_LEN + offsets_q
  tl.store(D_block_ptrs, D_block)


  
  
class FlashAttention(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q, K, V, is_causal, softmax_scale):
    # Q <- bs x nheads x seq_len x d
    # K <- bs x nheads x num_kvs x d
    # V <- bs x nheads x num_kvs x d
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    batch_size, num_heads, seq_len, emb_dim = Q.shape

    O = torch.empty_like(Q)

    # Split into BLOCK_SIZE groups for each head and each batch 
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_SIZE_Q']), batch_size * num_heads, 1)

    M = torch.empty(
      Q.shape[0:3], device=Q.device, dtype=torch.float32
    )

    stage = 3 if is_causal else 1

    _attention_forward[grid](
      Q=Q,
      K=K,
      V=V,
      softmax_scale=softmax_scale,
      M=M,
      O=O,
      stride_Q_batch=Q.stride(0),
      stride_Q_head=Q.stride(1),
      stride_Q_seq=Q.stride(2),
      stride_Q_dim=Q.stride(3),
      stride_K_batch=K.stride(0),
      stride_K_head=K.stride(1),
      stride_K_seq=K.stride(2),
      stride_K_dim=K.stride(3),
      stride_V_batch=V.stride(0),
      stride_V_head=V.stride(1),
      stride_V_seq=V.stride(2),
      stride_V_dim=V.stride(3),
      stride_O_batch=O.stride(0),
      stride_O_head=O.stride(1),
      stride_O_seq=O.stride(2),
      stride_O_dim=O.stride(3),
      BATCH_SIZE=Q.shape[0],
      NUM_HEADS=Q.shape[1],
      SEQ_LEN=Q.shape[2],
      EMB_DIM=Q.shape[3],
      STAGE=stage,
    )

    ctx.save_for_backward(Q, K, V, O, M)
    ctx.grid = grid
    ctx.softmax_scale = softmax_scale
    ctx.EMB_DIM = Q.shape[3]
    ctx.is_causal = is_causal
    return O
  
  @staticmethod
  def backward(ctx, dO):
    Q, K, V, O, M = ctx.saved_tensors

    assert dO.is_contiguous()
    assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
    NUM_WARPS, NUM_STAGES = 4, 3
    BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

    preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
    D = torch.empty_like(M) 

    # Compute Di - Intermediate value used for dK, dV, dQ
    _attention_backward_di[preprocess_grid](
        O=O,
        dO=dO,
        D=D,
        SEQ_LEN=SEQ_LEN,
        BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
        EMB_DIM=ctx.EMB_DIM,
    )

    grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

    stage = 3 if ctx.is_causal else 1

    # Fix KV, iterate Q to compute dK, dV
    _attention_backward_dk_dv[grid](
      Q=Q,
      K=K,
      V=V,
      softmax_scale=ctx.softmax_scale,
      dO=dO,
      dQ=dQ,
      dK=dK,
      dV=dV,
      M=M,
      D=D,
      stride_batch=Q.stride(0),
      stride_head=Q.stride(1),
      stride_seq=Q.stride(2),
      stride_dim=Q.stride(3),
      NUM_HEADS=NUM_HEADS,
      SEQ_LEN=SEQ_LEN,
      BLOCK_Q=BLOCK_SIZE_MICRO,
      BLOCK_KV=BLOCK_SIZE_MACRO,
      EMB_DIM=ctx.EMB_DIM,
      STAGE=stage,
      num_warps=NUM_WARPS,
      num_stages=NUM_STAGES,
    )

    _attention_backward_dq[grid](
        Q=Q,
        K=K,
        V=V,
        softmax_scale=ctx.softmax_scale,
        dO=dO,
        dQ=dQ,
        dK=dK,
        dV=dV,
        M=M,
        D=D,
        stride_batch=Q.stride(0),
        stride_head=Q.stride(1),
        stride_seq=Q.stride(2),
        stride_dim=Q.stride(3),
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        BLOCK_Q=BLOCK_SIZE_MACRO,
        BLOCK_KV=BLOCK_SIZE_MICRO,
        EMB_DIM=ctx.EMB_DIM,
        STAGE=stage,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return dQ, dK, dV, None, None