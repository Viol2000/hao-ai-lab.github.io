+++
title = "Efficient LLM Scheduling by Learning to Rank"
date = 2024-10-10T12:00:00-08:00
authors = ["Yichao Fu", "Siqi Zhu", "Runlong Su", "Aurick Qiao", "Ion Stoica", "Hao Zhang"]
author = "Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Stoica, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/vllm-ltr"
+++

{{< socialBadges arxiv-index="2408.15792" github="hao-ai-lab/vllm-ltr" >}}

**TL;DR:** Traditional Large Language Model (LLM) serving systems rely on first-come-first-serve (FCFS) scheduling. When longer requests block shorter ones in the queue, this creates a cascade of delays that severely impacts overall system latency. With output lengths being unpredictable in LLM generation, optimizing scheduling has remained challenging. We developed a novel *learning to rank* approach that predicts the relative ranking of output lengths, enabling a more efficient Shortest Job First-like scheduling policy. This scheduling approach reduced chatbot latency by 6.9x compared to traditional FCFS scheduling.

## Background

### Head-of-line Blocking in LLM Serving 

LLMs have become critical infrastructure for many Internet services, powering applications used by millions daily. As demand grows, serving LLMs efficiently on GPU clusters becomes essential to handle the sheer volume of user requests. For applications like chatbots, this requires minimizing user-perceived latency while maintaining high throughput to support as many concurrent users as possible.

Traditional first-come-first-serve (FCFS) scheduling, a common strategy in LLM serving, falls short under **high workloads** due to Head-of-Line (HOL) blocking. In FCFS, long-running requests monopolize the system, forcing shorter, potentially quicker requests to wait unnecessarily. This not only increases user-perceived latency but also hinders overall system efficiency. As illustrated in Figure 1, a single long request can significantly delay others, compounding performance bottlenecks.

It is well-established that algorithms like shortest-job-first (SJF) and the preemptive version shortest-remaining-time-first (SRTF) minimize the average latency, as they prioritize shorter tasks. However, SJF/SRTF are seldom implemented in LLM services because they require requests to be ordered by their remaining generation lengths, which is traditionally assumed to be difficult or impossible to know ahead of time in existing systems.

{{< image src="img/HOL.jpg" alt="HOL" width="120%" title="Figure 1: A long request can block short requests and introduce severe HOL blocking and high latency. We assume there is no prefill time, and the system takes 1 second to generate 1 token. With a First-come-first-serve (FCFS) schedule, the long request *R0*, which arrives first and takes 10 seconds to generate 10 tokens, will block subsequent shorter requests *R1* and *R2* for 10 seconds. Hence the latencies of *R0*,  *R1*, and *R2* are $10 / 10 = 1, (10 + 2) / 2 = 6, (10+2+1)/1=13 \mbox{ s / token}$, respectively, perceived by users, with an average latency of $(1+6+13)/3 = 6.67 \mbox{ s / token}$. By contrast, prioritizing shortest requests yields an average latency of $(1.3+1.5+1)/3=1.27 \mbox{ s / token}$ -- a $5.3\times$ reduction in average latency.">}}

### Learning to Rank

Learning to rank is a machine learning approach that learns to order items based on supervised ranking data. Among various ranking methods, we focus on listwise approaches that directly optimize the order of all items in a list. One such approach is [ListMLE](https://dl.acm.org/doi/10.1145/1390156.1390306), a listwise ranking loss function of particular interest in our paper.

ListMLE minimizes the likelihood function defined as $\mathcal{\phi}(g(x),y)=-\log P\left(y \mid x ; g\right)$, where

$P(y \mid x ; g)=\prod_{i=1}^n \frac{\exp \left(g\left(x_{y(i)}\right)\right)}{\sum_{k=i}^n \exp \left(g\left(x_{y(k)}\right)\right)} $

Here, $P(y \mid x ; g)$ represents the probability of permutation $y$ given input $x$ and scoring function $g$. $x_{y(i)}$ denotes the element in $x$ corresponding to the $i$-th position in permutation $y$. The idea is to maximize the likelihood of the correct ranking $y$ by using scoring function $g$ to predict the ranking of input $x$. The loss function $\mathcal{\phi}(g(x),y)$ minimizes the negative log-likelihood of this probability, encouraging the model to predict rankings close to the true ordering.

## LLM Scheduling by Learning-To-Rank

### Ranking is All You Need (to approximate SJF)

Due to LLM's autoregressive decoding, tokens are generated one by one, with each token depending on all previous tokens. Since we cannot predict when the model will generate an *End-Of-Sequence* token, precise generation lengths cannot be determined at the start. 

However, we demonstrate that exact lengths aren't necessary - accurate prediction of **generation length rankings** is sufficient.

Our goal is to approximate true SJF/SRTF scheduling using these rankings to reduce HOL blocking (Figure 2a) and achieve lower latency in LLM serving. As shown in Figure 2a, our approach achieves a normalized waiting time that's 0.5x of FCFS, while still being only 0.2x away from the optimal SRTF. Figure 2b demonstrates that higher Kendall's Tau correlations indicate more accurate rank predictions compared to the oracle (SJF/SRTF), which directly translates to lower latency per token in the LLM serving system.

{{< image src="img/ranking.png" alt="ranking" width="120%" title="Figure 2: (a): HOL blocking was evaluated by comparing FCFS and SRTF scheduling policies across 1K requests. (b): Analysis revealed that higher Kendall’s Tau correlation coefficients were associated with reduced latency. This finding was validated using the ShareGPT dataset with the Llama-3-8B model.">}}


### Method

We propose a straightforward yet effective algorithm for scheduling requests using ranking information (shown in Figure 3 and detailed in Algorithm 1 in the paper). The core idea is to:

- Iteratively run the predictor model ($P$) to score new requests
- Sort all requests by their predicted generation length rankings
- Form a running batch based on this sorted order, while respecting memory and batch size constraints 

To prevent long requests from being perpetually delayed, we've incorporated starvation prevention mechanisms, which we'll discuss after shortly. This ranking-based scheduler operates at the iteration level, making it compatible with established LLM serving techniques like [continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu) and [PagedAttention](https://dl.acm.org/doi/10.1145/3600006.3613165).

{{< image src="img/llm-ltr.png" alt="overview" width="80%" title="Figure 3: Overview of the method.">}}


### Training Length Ranking Predictor

For our predictor ($P$), we use a small [OPT](https://arxiv.org/abs/2205.01068) model (e.g., OPT-125M) that processes natural language prompts and generates ranking scores. While previous methods used classification with bucketing to predict output lengths, we found this approach both challenging and unnecessary - relative rankings are sufficient. Based on this insight, we train the OPT model using learning-to-rank techniques to order prompts by their expected generation length.

The training data consists of prompt-ranking pairs collected from actual serving batches. For each batch (e.g., size of 64), we record the prompts and their corresponding rankings based on their actual generation lengths (i.e., how many tokens were ultimately generated for each prompt). Once we've collected 10K such pairs from real-world serving, training the OPT model takes less than 5 minutes, making it practical for deployment.

### Starvation Prevention

While SJF/SRTF scheduling can improve overall latency, it risks causing starvation for long requests, where users wait excessively for responses. Unlike previous fairness designs that focus on [inter-client fairness](https://www.usenix.org/conference/osdi24/presentation/sheng), we propose a $max\_waiting\_time$ metric to evaluate fairness at the per-request level, directly reflecting individual user satisfaction. This metric considers both Time To First Token (TTFT) and Time Per Output Token (TPOT) in LLM serving:

$max\_waiting\_time = max(TTFT, max(TPOT))$

This metric characterizes the maximum time interval a user experiences between receiving tokens after submitting a request. A larger $max\_waiting\_time$ indicates longer waiting periods, signaling more severe starvation.

To mitigate starvation, our algorithm implements three mechanisms:
- Increment a request's starvation count when it isn't executed in a scheduling step
- Promote a request's priority by allocating "quantum" execution time once its starvation count reaches a threshold
- Maintain the elevated priority until the request exhausts its quantum

This approach prevents request-level starvation, improves max_waiting_time, and enhances user satisfaction, as our experiments (paper §5.5) demonstrate.


##  Experiments

### Main results

Figure 4 compares the latency of our ranking method against four baselines using two real-world datasets (ShareGPT and LMSYS-Chat-1M) across increasing arrival rates. We evaluated these methods on two latest models: LLaMA3 8B and 70B. At 64 requests/second, our method achieves up to 6.9x lower mean latency than FCFS and 1.5x-1.9x lower than Prompting Oracle (PO).

Both [Multi-Level Feedback Queue](https://arxiv.org/abs/2305.05920) (MLFQ, a traditional system scheduling approach) and [PO](https://dl.acm.org/doi/abs/10.5555/3666122.3668981) (which asks the LLM itself to predict its generation length) suffer from severe HOL blocking because they require initial processing of all requests: PO must run requests through the LLM, while MLFQ needs to process requests before assigning priority levels. The [classification approach](https://arxiv.org/abs/2306.06000), which sorts requests into length buckets, optimizes for high accuracy rather than ranking, missing opportunities for optimization.
 
{{< image src="img/main.png" alt="main.png" width="100%" title="Figure 4: Main results of LLM-Ltr">}}

### Overhead of the predictor

The predictor adds minimal overhead - less than 2% additional processing time across all settings. We use a 350M parameter predictor for the 70B model and a 125M predictor for the 8B model. While request processing involves both prefill and decode steps, the OPT predictor only performs prefill operations. The overhead increases with longer context lengths, which explains the higher overhead observed on the ShareGPT dataset.

{{< image src="img/overhead.png" alt="overhead" width="100%" title="Figure 5: Overhead of the predictor">}}

More detailed results can be found in the §5 of our paper.

## Get started

Please see [our paper](https://arxiv.org/abs/2408.15792) for more details. We also invite you to try out [our codebase](https://github.com/hao-ai-lab/vllm-ltr) and [checkpoints](https://huggingface.co/LLM-ltr/)!


## Acknowledgement

We extend our gratitude to Junda Chen, Yinmin Zhong, and Zhuohan Li for their valuable feedback!

## Citation

```
@article{fu2024efficient,
  title={Efficient LLM Scheduling by Learning to Rank},
  author={Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2408.15792},
  year={2024}
}
```
