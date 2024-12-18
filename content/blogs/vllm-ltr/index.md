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
[cover]
      image = "img/HOL.jpg"
      alt = "HOL"
      caption = "An illustration of Head-of-line."
+++

{{< socialBadges arxiv-index="2408.15792" github="hao-ai-lab/vllm-ltr" >}}

{{< justify >}}

**TL;DR:** Traditional Large Language Model (LLM) serving systems use first-come-first-serve scheduling since the exact output lengths are unpredictable. However, we developed a *learning to rank* approach that predicts the relative ranking of output lengths, enabling a more efficient scheduling policy that reduced chatbot latency by 2.8x and increased data generation throughput by 6.5x.

{{< /justify >}}




## Background: Head-of-line Blocking in LLM Serving 


In background, we introduce several background concepts in learning to rank, which are essential for understanding our methodology.


{{< image src="img/HOL.jpg" alt="HOL" width="120%" title="Figure 1: A long request can block short requests and introduce severe HOL blocking and high latency. We assume there is no prefill time, and the system takes 1 second to generate 1 token. With a First-come-first-serve (FCFS) schedule, the long request \textit{R0}, which arrives first and takes 10 seconds to generate 10 tokens, will block subsequent shorter requests \textit{R1} and \textit{R2} for 10 seconds. Hence the latencies of \textit{R0},  \textit{R1}, and \textit{R2} are $10 / 10 = 1, (10 + 2) / 2 = 6, (10+2+1)/1=13 \mbox{ s / token}$, respectively, perceived by users, with an average latency of $(1+6+13)/3 = 6.67 \mbox{ s / token}$. By contrast, prioritizing shortest requests yields an average latency of $(1.3+1.5+1)/3=1.27 \mbox{ s / token}$ -- a $5.3\times$ reduction in average latency.">}}

## Learning to Rank

{{< justify >}}
Learning to rank is a machine learning approach applied to supervised ranking data. It is widely used in recommendation systems, search engine and other research areas. Learning to rank typically takes one of three forms: pointwise, pairwise, and listwise. Pointwise turns the ranking problem into regression, classification or ordinal regression. Pairwise method learns the relative ranking for each pair of items. Listwise learns the ranking of lists of samples in a dataset. 
{{< /justify >}}

### ListMLE Loss

{{< justify >}}
ListMLE is a listwise ranking loss of particular interest in our paper. It minimizes the likelihood function defined $\mathcal{\phi}(g(x),y)=-\log P\left(y \mid x ; g\right)$, where

$P(y \mid x ; g)=\prod_{i=1}^n \frac{\exp \left(g\left(x_{y(i)}\right)\right)}{\sum_{k=i}^n \exp \left(g\left(x_{y(k)}\right)\right)} $   

Here, \( P(y \mid x ; g) \) represents the probability of the permutation \( y \) given the input \( x \) and the scoring function \( g \). \( x_{y(i)} \) denotes the element in \( x \) that corresponds to the \( i \)-th position in the permutation \( y \). The idea is to maximize the likelihood of the correct ranking \( y \) by using the scoring function \( g \) to predict the ranking of the input \( x \). The loss function \( \mathcal{\phi}(g(x),y) \) minimizes the negative log-likelihood of this probability, encouraging the model to predict a ranking close to the true ranking. ListMLE's focus on list ranking aligns with Kendall's Tau, which measures the correlation between two rankings. This ensures that minimizing the loss can help improve Kendall's Tau.
{{< /justify >}}

## LLM Scheduling by Learning-To-Rank

{{< justify >}}
We propose a simple but effective algorithm, for scheduling requests using ranking information, as detailed in Algorithm. The core idea is to iteratively run the predictor model $P$ to score new requests, then sort all requests according to their predicted generation length rankings. We form a running batch based on this sorted order, subject to memory or batch size constraints. To prevent the starvation of long requests, we've incorporated additional mechanisms, which we'll explain shortly. This ranking-based scheduling algorithm operates at the iteration level, making it compatible with established LLM serving techniques such as continuous batching and PagedAttention. 

{{< /justify >}}

##  Experiments

### Results
{{< justify >}}
Our experiments contain three domain-specific tasks, including Spider (text-to-SQL), Human-Eval (Python code completion), and GSM8k (math), and the broader open-domain conversational challenge, MT-bench. Reported experiments were conducted using either fine-tuned coder LLM, Deepseek-coder-7B-instruct, LLaMA-2-7B or ABEL-7B-001 as the target model depending on the task. Both training and evaluation are carried out on NVIDIA A100 40GB servers.
{{< /justify >}}

{{< image src="img/cllm_speedup.png" alt="speedup" width="70%" title="Figure 5: CLLM speedup on different downstream tasks. CLLMs are significantly faster than pre-trained models and achieve comparable speedups in comparison with Medusa, yet with no extra cost at inference time.">}}

{{< two_images src2="img/specialized_domains.png" src1="img/mt-bench.png" alt1="specialized" alt2="mt_bench" width1="50%" width2="50%" title="Figure 6: illustration of CLLM vs. other baselines on domain-specific tasks (Spider, CSN-Python, GSM8k), as well as on MT-bench. CLLMs achieve similar or even better speedup in comoparison with Medusa2 while introducing no extra inference cost (in terms FLOPS and memory consumption).">}}

{{< justify >}}
**Specialized domains:** From Figure 5, we can see that in comparison with other baselines including the original target model, Medusa2, and speculative decoding, CLLMs achieve the most significant speedup.

**Open-domain conversational Challenge (MT-bench):** CLLM trained from LLaMA2-7B using ShareGPT dataset can achieve roughly the same speedup as Medusa2 when combined with lookahead decoding, with comparable scores on MT-bench. However, CLLM offers higher adaptability and memory efficiency as it requires no modifications to the target model's original architecture and no auxiliary components.
{{< /justify >}}

### Training Cost 
{{< justify >}}
The fine-tuning cost of CLLMs is moderate, e.g., passing only around 1M tokens for LLaMA-7B to achieve a $3.4\times$ speedup on the Spider dataset. In the cases where the dataset size is large, for example, for CodeSearchNet-Python, only 10% of the dataset is required to generate Jacobi trajectories in training CLLMs to obtain around $2.5\times$ speedup. The total number of tokens can be estimated by taking:

$N = $ avg # of trajectories per prompt $ \times $ avg trajectory length $ \times $ # of prompts.
{{< /justify >}}

{{< center >}}
| dataset | estimated training cost (tokens) | $\%$ of pre-training cost
|:---:|:---:|:---:|
| Spider | 2M | $< 0.01\%$
| CodeSearchNet-Python | 100M | $\sim 0.1\%$
| GSM8K | 10M | $\sim 0.01\%$
| ShareGPT | 200M | $\sim 0.2\%$

{{< /center >}}

### Fast Forwarding and Stationary Tokens

{{< image src="img/trajectory_compare_aligned.png" alt="trajectory_compare" width="120%" title="Figure 7: Comparison of Jacobi trajectory between a target LLM and CLLMs on Spider. Each point along the Jacobi trajectory is a color-coded sequence: blue for correct tokens matching with AR results, and red for inaccurate ones. CLLM demonstrates enhanced efficiency, converging to the fixed point $2\times$ faster the Target LLM. This increased efficiency in the CLLM can be attributed to the consistency loss which facilitates the learning of the structure of each $n$-token sequence given a prefix.">}}

{{< justify >}}
The left side of Figure 6 shows target LLMs typically generate only one correct token in one iteration. In contrast, in CLLMs, we identify **fast forwarding phenomenon** where multiple consecutive tokens are correctly predicted in a single Jacobi iteration. 

Moreover, tokens correctly generated in advance (e.g. “country” and “H” at index 6 and 7 on the left side of Figure 6), are often replaced inaccurately in subsequent iterations in target LLMs. On the other hand, CLLMs exhibit the capability of predicting correct tokens preemptively, even with preceding incorrect tokens, while ensuring the tokens remain unchanged. We term such tokens as **stationary tokens**. Both phenomena contribute to the fast convergence in Jacobi decoding of CLLMs, thereby leading to a considerable generation speedup.

We observe that CLLMs acquire a crucial linguistic concept through training – **collocations**: a series of words or terms that [co-occur more frequently than one would expect by random chance](https://aclanthology.org/P91-1036.pdf). Language is not solely composed of isolated words but also relies heavily on specific word pairings. Examples of collocations are abundant in both natural and coding languages. They include verb + preposition combinations (e.g., ''talk to'', ''remind ... of ...''), verb + noun structures (e.g., ''make a decision'', ''catch a cold''), and many more domain-specific syntactical structures (e.g., ''SELECT ... FROM ...'', ''if ... else'' for programming). The consistency generation objective allows CLLMs to infer such structures from any point in the Jacobi trajectory, encouraging CLLMs to acquire proficiency in numerous collocations and thereby predict multiple words simultaneously to minimize iteration steps. 
{{< /justify >}}


## Get started
{{< justify >}}
Please see [our paper](https://arxiv.org/abs/2408.15792) for more details. We also invite you to try out [our codebase](https://github.com/hao-ai-lab/vllm-ltr) and [checkpoints](https://huggingface.co/LLM-ltr/)!
{{< /justify >}}

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
