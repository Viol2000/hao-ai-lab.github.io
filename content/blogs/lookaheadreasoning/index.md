+++
title = "Scaling Speculative Decoding with Lookahead Reasoning"
date = 2025-09-13T12:00:00-08:00
authors = ["Yichao Fu", "Yiming Zhao", "Rui Ge", "Hao Zhang"]
author = "Yichao Fu, Yiming Zhao, Rui Ge, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/LookaheadReasoning"
[cover]
      image = "/img/lr-acc-demo.gif"
      alt = "Lookahead Reasoning Demo"
      caption = "modify this caption"
+++

{{< socialBadges arxiv-index="2506.19830" >}}

{{< justify >}}

**TL;DR:** We propose **Lookahead Reasoning (LR)**, a technique that significantly accelerates large reasoning models(LRMs) and complements existing speculative decoding methods. Traditional token-level speculative decoding suffers from limited gains because the probability of correctly guessing a long sequence decreases exponentially with length. In contrast, LR operates at the step level, proposing future reasoning steps instead of individual tokens. This is much more effective since a proposed step only needs to be semantically correct, rather than matching exactly word for word. Importantly, LR is orthogonal to token-level approaches and can be combined with them to achieve multiplicative speedups. For example, on the AIME24 benchmark, our combined method increases the speedup from 1.4x to 2.1x without any loss in accuracy

{{< /justify >}}


## Background: Speedup of Speculative Decoding Is Upper-Bounded

Speculative decoding (SD) accelerates language model decoding by using a small drafter model to propose a sequence of future tokens (denoted by $\gamma$), which are then verified in parallel by the larger target model. If the entire $\gamma$-token sequence is accepted, the target model can process $\gamma$ tokens simultaneously in a single forward pass and advance its state by $\gamma + 1$ positions, bypassing the usual step-by-step autoregressive process.

However, the speedup achievable by token-level speculative decoding is fundamentally limited. Let $\alpha\in(0,1)$ denote the average per-token acceptance rate, $\gamma$ the number of drafted tokens, and $c$ the draft-to-target per-token latency ratio. Under the standard independence assumption, the expected number of target tokens validated in a single target forward pass is $1+\alpha+\cdots+\alpha^{\gamma}=\tfrac{1-\alpha^{\gamma+1}}{1-\alpha}$. The resulting expected wall-time speedup factor (Theorem 3.8 in [the speculative decoding paper](https://arxiv.org/abs/2211.17192)) is

$$
S(\alpha,\gamma,c)=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}.
$$


This formulation makes the fundamental limitation of the speedup explicit:
as $\gamma \to \infty$, the speedup saturates at the theoretical upper bound of $\frac{1}{1 - \alpha}$, regardless of how small the overhead $c$ is. Even in the idealized zero-overhead limit $c \to 0$, the speedup cannot exceed this bound. For any nonzero overhead ($c > 0$), increasing $\gamma$ further leads to diminishing returns, as the denominator $(\gamma c + 1)$ dominates. Thus, the speedup remains fundamentally capped.

This limitation is especially problematic for large reasoning models, which typically produce long, structured outputs with step-by-step logic. Since speculative decoding can only skip a small number of tokens at a time and this does not scale with the total output length, its contribution to end-to-end latency reduction becomes marginal for long-form reasoning tasks.



## Key Insight: Reasoning Happens in Steps, Not Just Tokens

Our key insight is that reasoning is hierarchical: a full chain-of-thought breaks into discrete steps. Crucially, these steps only require semantic correctness, not a perfect token-for-token match to reach a valid conclusion. This observation unlocks a far more powerful, coarser-grained approach to speculative decoding.

By shifting speculation from the token-level to the step-level, we mitigate the primary bottleneck of traditional SD. Instead of being constrained by the low probability of guessing long, exact token sequences, we can now speculatively generate and verify multiple semantically complete reasoning steps in parallel. Intuitively, successfully speculating a multi-token reasoning step, which only needs to be semantically correct to be accepted, should be more achievable than speculating a long sequence of tokens that must match exactly. Moreover, this step-level approach is complementary to existing methods; token-level speculation can still operate within each verified step, creating layered acceleration for enhanced overall speedup.

This leads to the central challenge: how does one actually perform step-level speculative decoding? At the token level, verification is straightforward: we compare the draft model's probability for the proposed token against the target model's probability and use rejection sampling to decide acceptance. But for step-level speculation, it's unclear how to determine whether a draft step aligns with the target model's distribution over the next reasoning step.

## Lookahead Reasoning: Verify Steps in a Correct Way

Our algorithm, **Lookahead Reasoning (LR)**, accelerates the generation of long-form reasoning by introducing a novel form of step-level parallelism. The process begins with a lightweight draft model that proactively generates several candidate future steps, which we can call $\{\hat{s}_1, \hat{s}_2, ...\}$.

Instead of processing these sequentially, the powerful target model takes all these drafted steps and performs a single, highly efficient **batched forward pass**. In this one pass, it generates its own "ground truth" versions of each step in parallel. The key to this process is that each target step $s_i$ is generated based on the prefix created by the *previous* drafted step, $\hat{s}_{i-1}$, allowing the model to explore multiple future reasoning paths at once.

Following this, a **semantic verifier** compares each draft step with its corresponding ground truth version, checking for semantic equivalence rather than a perfect textual match. The algorithm then accepts the entire sequence of correct drafts up until the first mismatch, appending the target's own generated step at that point of divergence.

The resulting speedup is significant. We effectively replace multiple slow, sequential calls to the target model with a single parallel operation. This allows us to generate and validate several correct reasoning steps for the latency cost of generating just one.
{{< image src="img/LookaheadReasoningStep.jpg" alt="LookaheadReasoning" width="100%" title="Figure 2: One cycle of Lookahead Reasoning. The draft model proposes $\gamma=3$ steps $\{\hat{s_1}$, $\hat{s_2}$, $\hat{s_3}\}$. The target model then generate $\{s_1$, $s_2$, $s_3\}$ based on prefixes and $\{\hat{s_1}$, $\hat{s_2}$, $\hat{s_3}\}$, respectively. Verifier checks if draft and target steps are semantically equivalent (e.g., $s_1 \approx  \hat{s_1}$). If the first two steps are equivalent but the third is not, Lookahead Reasoning outputs the verified draft steps ($\hat{s_1}$, $\hat{s_2}$) followed by the target's correction ($s_3$). This allows accepting multiple steps with only a lowered latency (e.g., $2t + T$) compared to the sequential target calls in autoregressive decoding (e.g., $3T$), where $t$ is draft step time and $T$ is target step time.">}}


### Verifier Selection

The choice of verifier ($V$) is a pivotal design consideration in LR. While an ideal semantic verifier ensures no accuracy loss, practical implementations face a primary trade-off between judgment precision and computational overhead;
Furthermore, the strictness of verification (e.g., a threshold) presents a secondary trade-off, potentially boosting draft acceptance and speedup at the risk of degrading task accuracy from erroneously accepted steps. We explore three common paradigms for semantic assessment (i.e., LLM-as-a-Judge for nuanced evaluation, embedding-based verifier for efficient similarity, and target model scoring) each with distinct cost-precision profiles. 

### Multi-Branch Drafting

To further increase the number of the accepted reasoning steps, we explore tree-structure generation where the draft model proposes multiple candidate steps at each speculative position. Specifically, instead of generating a single candidate chain, the draft $q$ can propose a set of $W$ alternative steps for each position $j$ in the draft sequence. Once a step is generated, the draft then proposes $W$ child  candidates in parallel for the subsequent position $j+1$. This branching process continues up to a maximum $\gamma$ steps, leading to an exponential growth in the total number of candidate sequences explores, i.e., $W^\gamma$. The target model $p$, however, still generate one single candidate continuation step for each position $j$ (based on the draft prefix). The verifier $V$ would then check if **any** of the $W$ proposed draft branches for that position $j$ semantically aligns with the target model's step. If such a match is found, that branch is accepted and other branches are discarded. This multi-branch strategy aims to boost the likelihood of speculative success, albeit at the cost of increased computational effort in the drafting phase.


## End-to-End Performance of Lookahead Reasoning

We evaluated the end-to-end performance of LR across diverse benchmarks using DeepSeek-R1-Distill and Qwen3 pairs. The detailed results are presented in Table 1. A key finding is LR's consistent ability to preseve task accuracy. Across a variety of benchmarks, LR's accuracy varies within a narrow range relative to the target model's autoregressive baseline, from approximately 1.0\% above to 2.1\% below baseline performance. This accuracy preservation contrasts with  SpecReason, which exhibited more noticeable accuracy reductions on several tasks (e.g., dropping from $91.8\%$ to $85.9\%$ on GSM8K with Deepseek-R1, a $\sim6\%$ decrease). This underscores LR's design principle of preserving output via robust semantic verification.

{{< image src="img/performance.png" alt="table" width="100%" title="Table 1: LR's Performance Across Datasets. Speedup is relative to the Autoregressive Decoding of the respective Target Model.">}}


Furthermore, LR achieves strong accuracy while maintaining high step acceptance rates, often above 50\% and reaching up to 63\%.
These substantial acceptance rates empirically support our initial insight that a smaller draft model can effectively predict semantically correct reasoning steps for a larger target model. LR also delivers significant efficiency gains. Its step-level parallelism is orthogonal to token-level speculative decoding, and their synergy produces substantial speedups. LR alone achieves speedups ranging from 1.04x to 1.71x across various benchmarks and model pairs. When combined with n-gram SD, the total speedup is further amplified, reaching up to 2.11x. This combined approach consistently outperforms n-gram SD alone, demonstrating the added value of step-level speculation. These results, consistent across both Deepseek-R1 and Qwen3 families, underscore the generalizable acceleration benefits of LR.

### Combining LR with SD

To empirically validate the orthogonality of LR with speculative decoding, we conducted experiments using prompt-lookup decoding (n-gram) on the AIME dataset. 

{{< image src="img/ablation.png" alt="combine" width="100%" title="Figure 3: Orthogonality of Lookahead Reasoning and Speculative Decoding. When used alone, the speedup from both LR and SD is limited by their draft length ($\gamma$).">}}

Figure 3 shows the orthogonality of LR and Speculative Decoding (SD). Subplot (a) shows that while LR alone with varying draft step number reaches a speedup around 1.4x, adding SD boosts this to approximately 1.9x. Similarly, subplot (b) illustrates that SD alone with varying Speculative Token Numbers peaks around 1.55x speedup, but combining it with LR again achieves up to 1.9×. Collectively, these results highlight that while either method in isolation offers limited gains, their combination consistently yields the most significant performance improvements, aligning with our theoretical analysis.

### Verifiers

We compare 
{{< image src="img/verifier.png" alt="combine" width="100%" title="Figure 3: Orthogonality of Lookahead Reasoning and Speculative Decoding. When used alone, the speedup from both LR and SD is limited by their draft length ($\gamma$).">}}

Figure 3 shows the orthogonality of LR and Speculative Decoding (SD). Subplot (a) shows that while LR alone with varying draft step number reaches a speedup around 1.4x, adding SD boosts this to approximately 1.9x. Similarly, subplot (b) illustrates that SD alone with varying Speculative Token Numbers peaks around 1.55x speedup, but combining it with LR again achieves up to 1.9×. Collectively, these results highlight that while either method in isolation offers limited gains, their combination consistently yields the most significant performance improvements, aligning with our theoretical analysis.


## Get Started with Lookahead Reasoning

We have implemented lookahead reasoning upon [vllm](https://github.com/vllm-project/vllm). Try to accelerate your LRM with [lookahead reasoning](https://github.com/hao-ai-lab/LookaheadDecoding)! 

## Citation

```
@article{fu2025scaling,
  title={Scaling Speculative Decoding with Lookahead Reasoning},
  author={Fu, Yichao and Ge, Rui and Shao, Zelei and Deng, Zhijie and Zhang, Hao},
  journal={arXiv preprint arXiv:2506.19830},
  year={2025}
}
```

