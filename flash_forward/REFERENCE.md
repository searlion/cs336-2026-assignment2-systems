Table of Contents
- [Motivation](#motivation)
- [What is Batched Inference?](#what)
- [Setup](#setup)
- [Scenario 1: UNBATCHED Inference (Sequential)](#unbatched)
- [Scenario 2: BATCHED Inference (Parallel)](#batched)
- [The Memory Bandwidth vs. Compute Trade-off](#tradeoff)
- [Code](#code)
- [Appendix A: The Problem of Uneven Sequence Lengths](#padding)
- [Appendix B: Two Tokens Per User — Is Masking Needed?](#two-tokens)

<a id="motivation">
## Motivation
</a>

Batching is a foundational optimization in Large Language Model (LLM) inference. Working through the math helps with:

* **Understanding hardware utilization:** GPUs are massively parallel calculators. Generating a single token for a single user leaves most of the GPU's compute units idle while waiting for the model weights to travel from HBM into the chip's compute units. Batching reduces this memory bandwidth bottleneck by amortizing the cost of loading model weights.
* **Debugging Performance:** Understanding batching explains why your inference servers can handle 10 concurrent users at almost the same speed as 1 user, and why batch size is constrained by KV cache memory.

---

<a id="what">
## What is Batched Inference?
</a>

In a production environment, an LLM server receives multiple requests from different users at the same time. 

Imagine User A asks: *"What is the capital of France?"*
Imagine User B asks: *"Write a poem about a cat."*

*Note: Both prompts get processed in one shot (the prefill phase), and then the server enters the decode phase: generating one token at a time for each user, until they're done. Decode is where almost all the time goes, and it's where batching earns its keep — because at each step each user contributes only a single token, which is too little work to keep a GPU busy on its own. Everything below describes one decode step.*

The model needs to predict the next token for User A and the next token for User B. 

If we don't batch, the server processes User A's token, and then processes User B's token sequentially. To do this, it must load the massive model weight matrices into the GPU's compute units (SRAM) to process User A, and then **load the exact same weights again** to process User B. 

**Batched inference** stacks the current token from User A and the current token from User B into a single mathematical matrix. By doing this, the model loads each weight matrix from HBM **exactly once** per forward pass instead of once per user, and multiplies it against both users' tokens simultaneously. 

*Note: "Once per forward pass" does not mean the weights get loaded once and stay put. On-chip SRAM holds tens of megabytes; the weights are tens of gigabytes. Each layer's weights are pulled in from HBM, used, and thrown away to make room for the next layer — so a single decoding step streams the whole model through the chip, and the next token does it all over again. Batching doesn't reduce that traffic. Batching just means more users get served per trip.*

### A note on memory hierachy

On an H100 (SXM variant) the tiers look like this:

| Tier | Size | Bandwidth | What lives there |
|---|---|---|---|
| On-chip SRAM (L1 / shared memory, per SM) | 256 KB × 132 SMs ≈ 33 MB | tens of TB/s, aggregate | the tile of weights and activations being multiplied *right now* |
| L2 cache | 50 MB | well above HBM, well below SRAM | whatever was touched recently |
| HBM3 | 80 GB | 3.35 TB/s | the model weights and every user's KV cache |

The memory system has to feed about 990 TFLOPS of BF16 on the tensor cores in order to fully utilize the SM cores, or 990 TFLOPS / 3.35 TB/s ≈ 295 FLOPs per byte. Any kernel that does fewer than roughly 300 floating-point operations per byte it pulls from HBM is waiting on memory. Any kernel that does more is waiting on the SM compute. That crossover is the "ridge point" from memory-bound to compute-bound.

For a batch-size of 1 decode step: it loads 2 bytes (BF16), then do 2 FLOPs, one multiply and one add. That's 1 FLOP per byte, about 300× short of the ridge. The tensor cores sit idle roughly 99.7% of the time, and the step takes as long as streaming the weights takes. For an 8-billion-parameter model in BF16 that's 16 GB / 3.35 TB/s ≈ 4.8 ms per token, a ceiling of about 210 tokens per second for a single user.

Batching increases the numerator, without changing the denominator. With N users in the batch each weight byte still crosses the bus once but now feeds N multiply-adds, so the weight matmuls run at roughly N FLOPs per byte. At N = 10 that's 30× under the ridge, which is why ten users cost about the same per step as one. The math only catches up with the memory somewhere around N ≈ 300, which is the "second, softer limit" from the trade-off section with a number attached. (In practice the crossover comes earlier because the KV cache also has to be read from HBM every step, and that traffic grows with both batch size and context length. While it is possible to disable KV cache during inference, memory use will still grow with longer context.)

Because a H100 only has 33 MB of SRAM against 16 GB of weights, the chip can hold about a five-hundredth of the model at any moment. Hence, in the course of a single decode step the on-chip memory fills and empties roughly 500 times to let every weight through once.

---

<a id="setup">
## Setup
</a>

We will walk through the calculation for predicting the next token for two different sequences. 

Assume our embedding dimension is {% katex inline %}d = 2{% endkatex %}. We have one weight matrix for our attention Queries: {% katex inline %}W_q{% endkatex %}. (We'll just focus on projecting the Query, but the exact same logic applies to Keys, Values, and the dense Feed-Forward layers).

Let our Query weight matrix be:
{% katex %} 
W_q = \begin{pmatrix} 1 & 1 \newline 0 & 1 \end{pmatrix}
{% endkatex %}

We have two users who have sent requests to our server. Both users are mid-conversation. Each has already had some tokens processed (their prompt, plus whatever has been generated so far), and the keys and values of those tokens sit in that user's KV cache. The scenarios below look only at the one *new* token per user that this step has to handle. The cached history comes back into play at attention, in the BONUS section of the code, where we'll assume it happens to be 3 tokens per user.
* **User A** has a sequence that just produced a new token, {% katex inline %}x_A{% endkatex %}.
* **User B** has a completely unrelated sequence that just produced a new token, {% katex inline %}x_B{% endkatex %}.

Let their current token embeddings be:
* {% katex inline %}x_A = [1, 0]{% endkatex %}
* {% katex inline %}x_B = [0, 1]{% endkatex %}

We need to calculate the new Query vector for both users so they can proceed with their respective attention mechanisms.

---

<a id="unbatched">
## Scenario 1: UNBATCHED Inference (Sequential)
</a>

Without batching, the server processes the requests sequentially (a batch size of 1). 

### Step 1: Process User A
The GPU loads the weight matrix {% katex inline %}W_q{% endkatex %} from its High Bandwidth Memory (HBM) into the small, fast on-chip SRAM that the compute cores read from. It performs a Vector-Matrix multiplication for User A.

{% katex %} 
q_A = x_A \times W_q = \begin{pmatrix} 1 & 0 \end{pmatrix} \times \begin{pmatrix} 1 & 1 \newline 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \end{pmatrix}
{% endkatex %}

### Step 2: Process User B
The GPU, having finished with User A, now moves to User B. It must load the weight matrix {% katex inline %}W_q{% endkatex %} from HBM onto the chip *all over again*.

{% katex %} 
q_B = x_B \times W_q = \begin{pmatrix} 0 & 1 \end{pmatrix} \times \begin{pmatrix} 1 & 1 \newline 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \end{pmatrix}
{% endkatex %}

We got our answers, but we loaded the weights twice to do very little math.

---

<a id="batched">
## Scenario 2: BATCHED Inference (Parallel)
</a>

Now, let's process the requests together. We take User A's token and User B's token and stack them vertically into a single Input Batch Matrix, {% katex inline %}X_{batch}{% endkatex %}.

{% katex %} 
X_{batch} = \begin{pmatrix} 1 & 0 \newline 0 & 1 \end{pmatrix}
{% endkatex %}
*(Row 1 is User A, Row 2 is User B)*

### Step 1: Process the Batch
The GPU loads the weight matrix {% katex inline %}W_q{% endkatex %} from HBM **once**. It performs a Matrix-Matrix multiplication.

{% katex %} 
Q_{batch} = X_{batch} \times W_q = \begin{pmatrix} 1 & 0 \newline 0 & 1 \end{pmatrix} \times \begin{pmatrix} 1 & 1 \newline 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \newline 0 & 1 \end{pmatrix}
{% endkatex %}

Notice the output:
* Row 1 is {% katex inline %}[1, 1]{% endkatex %}, which is exactly {% katex inline %}q_A{% endkatex %}.
* Row 2 is {% katex inline %}[0, 1]{% endkatex %}, which is exactly {% katex inline %}q_B{% endkatex %}.

We got the exact same answers for both users, but we only had to load the model weights from memory one time which is dominated by fetching `W_q`, not by the multiply itself. The extra row of math rides along for free in time the GPU was going to spend waiting anyway. That's why doubling the batch here costs almost nothing. A property that holds until the batch gets large enough to saturate the compute cores (see the trade-off section below).

---

<a id="tradeoff">
## The Memory Bandwidth vs. Compute Trade-off
</a>

In LLM inference (specifically during the decoding phase, when generating tokens one by one), the bottleneck is **Memory Bandwidth**, i.e. the time it takes to move the gigabytes of model weights from the GPU's HBM into the small, fast on-chip SRAM that feeds the processing cores, for every single token step.

* **Without Batching (Batch Size = 1):** The GPU spends a significant portion of time waiting for weights to travel across the memory bus. Once the weights arrive, the processing cores do a tiny Vector-Matrix calculation instantly, throw the weights away, and wait for the next load. The compute cores are starving.
* **With Batching (Batch Size = N):** You load the weights once, but now the compute cores do {% katex inline %}N{% endkatex %} times as much math (Matrix-Matrix multiplication). You are using "free" compute time that would have otherwise been spent waiting for memory. 

**The Limit to Batching**

While the model *weights* are shared across all users, every single user requires their own dedicated KV Cache to store their unique sequence history (as discussed in the KV Cache article). If you increase the batch size too much, the combined KV caches of all those users will exceed your GPU's total HBM, resulting in an Out Of Memory (OOM) error. Even without KV cache during inference, longer context will still demand more memory.

As the batch grows, you eventually give the compute cores more math than they can hide behind the weight loading. Once you cross that point, the GPU is compute-bound: doubling the batch roughly doubles the time of each decoding step, so total throughput flattens out while every individual user waits longer for each token. We can therefore cap batch size based on whichever comes first — hitting the HBM limit, or reaching the maximum per-token latency promised to users.

---

<a id="code">
## Code
</a>

Below is the PyTorch implementation showing how independent vectors are stacked into a batch. Note the use of `torch.bmm` (Batched Matrix Multiplication) to handle the attention step where every sequence in the batch has its own independent KV cache.

```python
import torch

# 1. SETUP
# Weight matrix (2x2)
W_q = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

# Individual tokens for User A and User B
x_A = torch.tensor([[1.0, 0.0]]) # Shape: 1x2
x_B = torch.tensor([[0.0, 1.0]]) # Shape: 1x2

print("=== SCENARIO 1: UNBATCHED ===")
# Streamed W_q from HBM onto the chip (Wait...)
q_A = x_A @ W_q
# Streamed W_q from HBM onto the chip AGAIN (Wait...)
q_B = x_B @ W_q

print(f"User A Query: {q_A}")
print(f"User B Query: {q_B}\n")

print("=== SCENARIO 2: BATCHED ===")
# Stack tokens into a batch
X_batch = torch.cat([x_A, x_B], dim=0) # Shape: 2x2
print(f"X_batch shape: {X_batch.shape}")

# Load W_q ONCE and multiply
Q_batch = X_batch @ W_q
print(f"Q_batch shape: {Q_batch.shape}")
print(f"Batched Queries:\n{Q_batch}\n")

# Verify mathematically
assert torch.allclose(q_A, Q_batch[0:1])
assert torch.allclose(q_B, Q_batch[1:2])
print("SUCCESS: Batched output perfectly matches sequential output!\n")

print("=== BATCHED ATTENTION (BONUS) ===")
# Each user has their own history, so the two K caches are stacked along a
# batch dimension into one 3D tensor: (Batch, Seq_Len, Dim). Stacking needs
# both histories to be the same length; assume 3 tokens each (Appendix A
# covers what happens when they aren't).
K_hist = torch.tensor([
    [[1., 0.], [1., 1.], [0., 1.]],  # User A's history (3x2)
    [[0., 1.], [2., 0.], [1., 1.]]   # User B's history (3x2)
])

# The new token's own key joins the cache before attention: a token attends
# to its history AND to itself. Take W_k = I so the key is just the embedding
# (the K projection is a per-row matmul exactly like W_q).
W_k = torch.eye(2)
K_new = (X_batch @ W_k).unsqueeze(1)             # (2, 2) -> (2, 1, 2)
K_cache = torch.cat([K_hist, K_new], dim=1)      # (2, 4, 2): 3 history + 1 new

Q_3D = Q_batch.unsqueeze(1)                      # (2, 2) -> (2, 1, 2)

# bmm multiplies User A's Q with User A's K and User B's Q with User B's K.
# K_cache is transposed on its last two dims: (2, 4, 2) -> (2, 2, 4).
# (Real implementations divide by sqrt(d), then softmax and multiply by V; omitted.)
attention_scores = torch.bmm(Q_3D, K_cache.transpose(1, 2))

print(f"Attention Scores Shape: {attention_scores.shape} -> (Batch, 1, Hist + 1)")
print(f"Scores:\n{attention_scores}")
```

Output:
```plaintext
=== SCENARIO 1: UNBATCHED ===
User A Query: tensor([[1., 1.]])
User B Query: tensor([[0., 1.]])

=== SCENARIO 2: BATCHED ===
X_batch shape: torch.Size([2, 2])
Q_batch shape: torch.Size([2, 2])
Batched Queries:
tensor([[1., 1.],
        [0., 1.]])

SUCCESS: Batched output perfectly matches sequential output!

=== BATCHED ATTENTION (BONUS) ===
Attention Scores Shape: torch.Size([2, 1, 4]) -> (Batch, 1, Hist + 1)
Scores:
tensor([[[1., 2., 1., 1.]],

        [[1., 0., 1., 1.]]])
```

---
<a id="padding">
## Appendix A: The Problem of Uneven Sequence Lengths
</a>

The toy example above assumes a perfect world on two counts. User A and User B both produced a new token at the exact same moment, so they could share a decode step. And in the BONUS section, both happened to have exactly 3 tokens of history in their KV cache, so the two caches stacked into one rectangular tensor.

In reality, batched matrices must be perfectly rectangular. What happens if User A's prompt is 100 tokens long, but User B's prompt is only 5 tokens long? 

**The Padding Problem**
Historically, the solution was **padding**. You would add 95 empty "pad" tokens to User B's cache just to make it the same shape as User A's cache so they could fit into a single `(2, 100, d)` 3D tensor, and specifically left-padding, with the pads in front, since a decoder generates from whatever sits in the last position and right-padding would leave the model staring at a pad token instead of User B's actual final token.

This is disastrous for efficiency. The GPU wastes precious FLOPs multiplying queries against padding tokens (which are then masked out and ignored anyway), and wastes precious HBM storing padding tokens in the KV cache. It is also a persistent source of bugs because the attention mask and the position IDs both have to be shifted to match the padding, and getting either wrong produces output that looks plausible but is subtly incorrect.

Three major innovations solved this:
1. **Wasted FLOPs in the kernel — ragged / varlen attention (FlashAttention, xFormers)**: these solve padding inside a single forward pass. Instead of demanding a rectangular (Batch, Seq_Len, Dim) tensor, they take all tokens flattened into one long sequence plus an index of where each sequence starts, and never touch a padding token.
(Concretely: the tokens are flattened into one long buffer, and the kernel is handed a list of cumulative offsets — [0, 100, 105] for a 100-token and a 5-token sequence, telling it where each sequence begins and ends.)
2. **Wasted HBM in the cache — PagedAttention (vLLM)**: Padding's storage-side cousin: rather than reserving one contiguous max-length block of HBM per user, the cache is allocated in small fixed-size pages on demand, like virtual memory in an operating system. A 5-token user occupies 5 tokens' worth of pages, not 100, and the gaps between users' blocks stop fragmenting the rest. This is what makes the KV cache limit from the trade-off section a limit on tokens actually in flight rather than on worst-case reservations.
3. **Wasted batch slots over time — continuous batching (iteration-level scheduling)**: Prompts aren't the only thing that varies in length; generations do too. In a static batch, when User B finishes after 10 tokens their seat sits empty (or worse, keeps generating pad tokens) while User A grinds through 500 more. Continuous batching ejects finished requests and admits new ones between token steps, so the batch stays full.

---
<a id="two-tokens">
## Appendix B: Two Tokens Per User — Is Masking Needed?
</a>

Everything so far has been one decode step: each user contributes exactly one new token. But there are common situations where a user contributes *several* tokens to the same forward pass — prefill (a prompt is many tokens), chunked prefill, and speculative decoding (verifying {% katex inline %}k{% endkatex %} draft tokens at once). The smallest version of that situation: **two users, two new tokens each**.

Does batching still work? And now that each user's tokens sit next to each other in the same matrix, do we need a mask to keep them from interfering?

**The projections need no mask at all. Attention needs one, but only the causal mask inside each user's own sequence.** Keeping User A away from User B costs nothing extra as long as we keep the batch dimension. 

### Setup

Keep {% katex inline %}W_q{% endkatex %} from before. User A's first new token is the same {% katex inline %}[1, 0]{% endkatex %} as before and User B's is the same {% katex inline %}[0, 1]{% endkatex %}; each now has a second token behind it:

* **User A:** {% katex inline %}x_{A,1} = [1, 0]{% endkatex %}, {% katex inline %}x_{A,2} = [1, 1]{% endkatex %}
* **User B:** {% katex inline %}x_{B,1} = [0, 1]{% endkatex %}, {% katex inline %}x_{B,2} = [2, 0]{% endkatex %}

So each user contributes two new tokens, and the batch as a whole contains four. Both counts show up below: "two" whenever we're looking at one user (the 2 rows of a user's score block, the 2 new keys appended to a cache), "four" whenever we're looking at the whole batch (the 4 rows of the projection in Step 1, the 4 query rows of the grid in Step 2).

### Step 1: The projection — no mask

Stack all four rows into one matrix and multiply, exactly as in Scenario 2:

{% katex %}
Q = \begin{pmatrix} 1 & 0 \newline 1 & 1 \newline 0 & 1 \newline 2 & 0 \end{pmatrix} \times \begin{pmatrix} 1 & 1 \newline 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \newline 1 & 2 \newline 0 & 1 \newline 2 & 2 \end{pmatrix}
{% endkatex %}

Row 1 is {% katex inline %}q_{A,1} = [1, 1]{% endkatex %} — the same {% katex inline %}q_A{% endkatex %} we have now computed three times. Row 2 is {% katex inline %}q_{A,2}{% endkatex %}, row 3 is {% katex inline %}q_{B,1}{% endkatex %}, row 4 is {% katex inline %}q_{B,2}{% endkatex %}.

A projection is a per-row operation. Row 2 is computed from {% katex inline %}x_{A,2}{% endkatex %} and {% katex inline %}W_q{% endkatex %} and nothing else; it cannot see rows 1, 3, or 4. The matrix has no idea whether it holds four users with one token each, one user with four tokens, or two users with two. So the memory-bandwidth argument from Scenario 2 carries over unchanged — {% katex inline %}W_q{% endkatex %} is loaded from HBM once and amortized over four rows instead of two — and there is nothing to mask.

The same is true of the K and V projections and every feed-forward layer. That's most of the model's weights, and none of them care who the rows belong to.

Notice that the history tokens don't appear in this step at all, and nothing is missing. Their projections were computed in the steps that produced them: their queries were used once and thrown away, and their keys and values went into the cache. This step projects only the four embeddings that are new, and the same multiplication with {% katex inline %}W_k{% endkatex %} and {% katex inline %}W_v{% endkatex %} yields the four new keys and values that get appended to the caches in Step 3. That is the division of labour in a decode step: the weight matrices only ever see new tokens, and the history reaches the computation exclusively through the cache at attention. It's why the projection cost of a step grows with the number of new tokens, not with the length of the conversation so far.

### Step 2: Attention — where users and positions start to matter

Attention is the one place a token interacts with *other* tokens, so it's the one place where "which user?" and "which position?" matter. There are two things to prevent:

1. **Cross-user leakage:** {% katex inline %}q_{A,2}{% endkatex %} must not attend to any of User B's keys.
2. **Future leakage:** {% katex inline %}q_{A,1}{% endkatex %} must not attend to {% katex inline %}k_{A,2}{% endkatex %}, which comes *after* it in User A's sequence. A decoder only looks backward.

If we flattened the four new tokens into one long sequence and naively let every query score every key, we'd get a grid of scores, one per (query, key) pair: 4 rows for the new queries, and 10 columns for the keys, because each user's cache holds 3 history keys plus the 2 new ones. Not all of those pairs are legitimate. Each cell asks: may this query (row) look at this key (column)? Two rules decide. The key must belong to the same user, and it must not come later than the query in that user's sequence (a token may see itself). Apply both and the allowed pairs look like this:

```plaintext
              User A's keys           User B's keys
             hist      new           hist      new
            k0 k1 k2   k3 k4        k0 k1 k2   k3 k4
A1 (q3)   [  1  1  1    1  0    |    0  0  0    0  0  ]
A2 (q4)   [  1  1  1    1  1    |    0  0  0    0  0  ]
          ---------------------+----------------------
B1 (q3)   [  0  0  0    0  0    |    1  1  1    1  0  ]
B2 (q4)   [  0  0  0    0  0    |    1  1  1    1  1  ]
```

Read row 1: {% katex inline %}q_{A,1}{% endkatex %}, sitting at position 3, may see User A's three history keys (all in its past) and its own key, but not {% katex inline %}k_{A,2}{% endkatex %} at position 4, and nothing of User B's. Row 4: {% katex inline %}q_{B,2}{% endkatex %} may see all five of User B's keys and none of User A's.

The two rules leave different fingerprints. "Same user" zeroes the two off-diagonal blocks. "Not in the future" zeroes the upper triangle of the new-token columns inside each diagonal block; the history columns are untouched by it, because history is always in the past. The two rules also have different fixes.

It is not efficient to compute the entire 4×10 grid and masks it because the two off-diagonal 2×5 blocks are wasted work. There are two ways to avoid computing them, and they differ only in who enforces the block structure. Either way, what gets computed is exactly the two diagonal 2×5 blocks, and the top-left one is the score matrix we'll work through in Step 3.

**The first is to keep the batch dimension**. Instead of a (4, d) matrix, hold the new tokens as a 3D tensor of shape (Batch, Seq_New, Dim) = (2, 2, 2), and the KV caches as (2, 5, 2), since each user now has 3 history keys plus 2 new ones. `torch.bmm` multiplies User A's queries against User A's keys and User B's against User B's; the cross-user blocks are unreachable by construction. The cost is that everything must be rectangular, which is the padding problem of Appendix A.

Concretely: a tensor is a box, and every slice along a dimension has to be the same length. The middle dimension of `(2, 5, 2)` is one number, so it can only be 5 if *both* users have exactly 5 keys. Take one history token away from User B, so they have 2 + 2 = 4 keys while User A still has 3 + 2 = 5, and the stack refuses:

```python
K_A = torch.tensor([[1., 0.], [1., 1.], [0., 1.], [1., 0.], [1., 1.]])  # 3 history + 2 new = 5 keys
K_B = torch.tensor([[0., 1.], [2., 0.], [0., 1.], [2., 0.]])            # 2 history + 2 new = 4 keys

torch.stack([K_A, K_B])
# RuntimeError: stack expects each tensor to be equal size,
#               but got [5, 2] at entry 0 and [4, 2] at entry 1
```

The only way into a `(2, 5, 2)` tensor is to give User B a fifth key that doesn't exist: a row of zeros, placed at the front (left-padding, for the reason given in Appendix A).

```python
K_B_padded = torch.cat([torch.zeros(1, 2), K_B], dim=0)   # now 5 x 2
K_cache = torch.stack([K_A, K_B_padded])                  # (2, 5, 2) -- works
scores = torch.bmm(Q_batch, K_cache.transpose(1, 2))      # Q_batch: the (2, 2, 2) queries from Step 1
```

`bmm` runs now, but User B's scores have a column for a key that isn't real:

```plaintext
User B raw scores:
[[0., 1., 0., 1., 0.],
 [0., 2., 4., 2., 4.]]
   ^ pad column
```

That 0 is not "ignored". It is a score like any other, and softmax would hand the pad key a weight of {% katex inline %}e^0{% endkatex %} divided by the row sum. So the pad forces a second kind of masking on top of the causal one, and it changes the mask's shape: User A's rows are causal-only, User B's are pad-plus-causal, so the mask becomes a per-user `(2, 2, 5)` rather than a single `(2, 5)` that broadcasts across the batch. The pad column also costs FLOPs to compute and HBM to store, for a key that is thrown away the moment it is scored. Multiply by real prompt lengths, 100 tokens versus 5, and you are looking at Appendix A's problem.

**The second method is to flatten to a ragged layout and hand the kernel two offset lists**: where each user's queries start in the flat query buffer, `cu_seqlens_q = [0, 2, 4]`, and where each user's keys start in the flat KV buffer, `cu_seqlens_k = [0, 5, 10]`. Those are precisely the row groups and column groups of the grid above.

Either way, user isolation falls out of the layout. What does not fall out of the layout is the causal triangle inside each block: `bmm` (or the kernel) happily computes q_{A,1} · k_{A,2}.

**The causal mask has to be explicit.** Inside User A's block, `bmm` happily computes {% katex inline %}q_{A,1} \cdot k_{A,2}{% endkatex %}, and nothing in the tensor layout says that pair is illegal. We have to say so.

### Step 3: The numbers

Take {% katex inline %}W_k = I{% endkatex %} so a new token's key is just its embedding (any {% katex inline %}W_k{% endkatex %} works; this keeps the arithmetic small). User A's keys are the three history keys from the BONUS section plus the two new ones:

{% katex %}
K_A = \begin{pmatrix} 1 & 0 \newline 1 & 1 \newline 0 & 1 \newline \mathbf{1} & \mathbf{0} \newline \mathbf{1} & \mathbf{1} \end{pmatrix}, \qquad
Q_A = \begin{pmatrix} 1 & 1 \newline 1 & 2 \end{pmatrix}
{% endkatex %}

Scores for User A — one row per new query, one column per key:

{% katex %}
Q_A K_A^T = \begin{pmatrix} 1 & 2 & 1 & 1 & 2 \newline 1 & 3 & 2 & 1 & 3 \end{pmatrix}
{% endkatex %}

The first three entries of row 1, {% katex inline %}[1, 2, 1]{% endkatex %}, are exactly the scores from the BONUS section: same query, same history. The two new columns are the new keys. The top-right entry, {% katex inline %}q_{A,1} \cdot k_{A,2} = 2{% endkatex %}, is the illegal one: token 1 peeking at token 2. Apply the causal mask:

{% katex %}
\begin{pmatrix} 1 & 2 & 1 & 1 & -\infty \newline 1 & 3 & 2 & 1 & 3 \end{pmatrix}
{% endkatex %}

After softmax, {% katex inline %}e^{-\infty} = 0{% endkatex %}, so the weight on that key is exactly zero. Every history column stays visible to both rows (history is always in the past), and the diagonal of the new block stays visible (a token may attend to itself). Because both users have the same amount of history, the mask is a single (2, 5) boolean that broadcasts across the batch dimension.

### Code

```python
import torch

# 1. SETUP
W_q = torch.tensor([[1.0, 1.0], [0.0, 1.0]])   # same W_q as before
W_k = torch.eye(2)                              # identity keeps the arithmetic small

# Each user now has TWO new tokens. Shape: (Batch, Seq_New, Dim) = (2, 2, 2)
X_batch = torch.tensor([
    [[1., 0.], [1., 1.]],   # User A: token 1, token 2
    [[0., 1.], [2., 0.]],   # User B: token 1, token 2
])

# 2. PROJECTIONS: no mask. A projection is a per-row operation.
Q_batch = X_batch @ W_q     # (2, 2, 2) — matmul broadcasts over the batch dim
K_new   = X_batch @ W_k     # (2, 2, 2)

# Flattening all four tokens into one (4, 2) matrix gives identical rows.
assert torch.allclose(Q_batch, (X_batch.reshape(4, 2) @ W_q).reshape(2, 2, 2))
print("Q_batch (Batch, Seq_New, Dim):")
print(Q_batch, end="\n\n")

# 3. KV CACHE: the same 3 tokens of history as before, extended by the 2 new keys.
K_hist = torch.tensor([
    [[1., 0.], [1., 1.], [0., 1.]],  # User A's history
    [[0., 1.], [2., 0.], [1., 1.]],  # User B's history
])
K_cache = torch.cat([K_hist, K_new], dim=1)   # (2, 5, 2)
print(f"K_cache shape: {K_cache.shape} -> (Batch, Hist + Seq_New, Dim)\n")

# 4. SCORES: bmm keeps users apart. User A's queries never meet User B's keys.
scores = torch.bmm(Q_batch, K_cache.transpose(1, 2))   # (2, 2, 5)
print("Raw scores (Batch, Seq_New, Hist + Seq_New):")
print(scores, end="\n\n")

# 5. CAUSAL MASK: the one place masking is needed.
# New token i may see all history and new tokens 0..i, but nothing after it.
hist_len, new_len = K_hist.shape[1], X_batch.shape[1]
mask = torch.ones(new_len, hist_len + new_len, dtype=torch.bool)
mask[:, hist_len:] = torch.tril(torch.ones(new_len, new_len, dtype=torch.bool))
print("Causal mask (True = may attend):")
print(mask, end="\n\n")

masked_scores = scores.masked_fill(~mask, float("-inf"))   # broadcasts over batch
print("Masked scores:")
print(masked_scores, end="\n\n")

weights = torch.softmax(masked_scores, dim=-1)
print("Attention weights (top-right of each user is exactly 0):")
print(weights)
```

Output:
```plaintext
Q_batch (Batch, Seq_New, Dim):
tensor([[[1., 1.],
         [1., 2.]],

        [[0., 1.],
         [2., 2.]]])

K_cache shape: torch.Size([2, 5, 2]) -> (Batch, Hist + Seq_New, Dim)

Raw scores (Batch, Seq_New, Hist + Seq_New):
tensor([[[1., 2., 1., 1., 2.],
         [1., 3., 2., 1., 3.]],

        [[1., 0., 1., 1., 0.],
         [2., 4., 4., 2., 4.]]])

Causal mask (True = may attend):
tensor([[ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]])

Masked scores:
tensor([[[1., 2., 1., 1., -inf],
         [1., 3., 2., 1., 3.]],

        [[1., 0., 1., 1., -inf],
         [2., 4., 4., 2., 4.]]])

Attention weights (top-right of each user is exactly 0):
tensor([[[0.1749, 0.4754, 0.1749, 0.1749, 0.0000],
         [0.0513, 0.3790, 0.1394, 0.0513, 0.3790]],

        [[0.2969, 0.1092, 0.2969, 0.2969, 0.0000],
         [0.0414, 0.3057, 0.3057, 0.0414, 0.3057]]])
```
