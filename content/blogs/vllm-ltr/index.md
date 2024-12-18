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

**TL;DR:** Traditional Large Language Model (LLM) serving systems use first-come-first-serve scheduling since the exact output lengths are unpredictable. However, we developed a *learning to rank* approach that predicts the relative ranking of output lengths, enabling a more efficient scheduling policy (to approximate SJF) that reduced chatbot latency by 2.8x and increased data generation throughput by 6.5x.

{{< /justify >}}

## Background

### Head-of-line Blocking in LLM Serving 

LLMs are increasingly becoming the backbone of many today’s Internet services and applications that serve millions of users. Due to the surge in demand, efficient scheduling for LLM serving is crucial to ensure high-quality service amidst numerous concurrent users competing for computing resources. For popular interactive applications such as chatbots, this means minimizing the latency that each user perceives while maximizing the overall system throughput to accommodate as many users as possible.

**Under high workload**, LLM services that implement a first-come-first-serve (FCFS) scheduling strategy inevitably face significant Head-Of-Line (HOL) blocking, as many requests must wait for others to execute. Figure 1 illustrates a typical example of how a long request can block shorter ones in FCFS scheduling, leading to significant HOL blocking. In such scenarios, it is well-established that the shortest-job-first (SJF) and shortest-remaining-time-first (SRTF) scheduling algorithms minimize the average latency experienced across all requests. However, SJF/SRTF are seldom implemented in LLM services because they require requests to be ordered by their **remaining generation lengths**, which is traditionally assumed to be difficult or impossible to know ahead of time in existing systems.

{{< image src="img/HOL.jpg" alt="HOL" width="120%" title="Figure 1: A long request can block short requests and introduce severe HOL blocking and high latency. We assume there is no prefill time, and the system takes 1 second to generate 1 token. With a First-come-first-serve (FCFS) schedule, the long request *R0*, which arrives first and takes 10 seconds to generate 10 tokens, will block subsequent shorter requests *R1* and *R2* for 10 seconds. Hence the latencies of *R0*,  *R1*, and *R2* are $10 / 10 = 1, (10 + 2) / 2 = 6, (10+2+1)/1=13 \mbox{ s / token}$, respectively, perceived by users, with an average latency of $(1+6+13)/3 = 6.67 \mbox{ s / token}$. By contrast, prioritizing shortest requests yields an average latency of $(1.3+1.5+1)/3=1.27 \mbox{ s / token}$ -- a $5.3\times$ reduction in average latency.">}}

### Learning to Rank

{{< justify >}}
Learning to rank is a machine learning approach applied to supervised ranking data. It is widely used in recommendation systems, search engine and other research areas. Learning to rank typically takes one of three forms: pointwise, pairwise, and listwise. Pointwise turns the ranking problem into regression, classification or ordinal regression. Pairwise method learns the relative ranking for each pair of items. Listwise learns the ranking of lists of samples in a dataset. 
{{< /justify >}}

{{< justify >}}
ListMLE is a listwise ranking loss of particular interest in our paper. It minimizes the likelihood function defined $\mathcal{\phi}(g(x),y)=-\log P\left(y \mid x ; g\right)$, where

$P(y \mid x ; g)=\prod_{i=1}^n \frac{\exp \left(g\left(x_{y(i)}\right)\right)}{\sum_{k=i}^n \exp \left(g\left(x_{y(k)}\right)\right)} $   

Here, \( P(y \mid x ; g) \) represents the probability of the permutation \( y \) given the input \( x \) and the scoring function \( g \). \( x_{y(i)} \) denotes the element in \( x \) that corresponds to the \( i \)-th position in the permutation \( y \). The idea is to maximize the likelihood of the correct ranking \( y \) by using the scoring function \( g \) to predict the ranking of the input \( x \). The loss function \( \mathcal{\phi}(g(x),y) \) minimizes the negative log-likelihood of this probability, encouraging the model to predict a ranking close to the true ranking. ListMLE's focus on list ranking aligns with Kendall's Tau, which measures the correlation between two rankings. This ensures that minimizing the loss can help improve Kendall's Tau.
{{< /justify >}}

## LLM Scheduling by Learning-To-Rank

### Ranking is All You Need (to appx. SJF)

We show that the precise generation length is not needed. An accurate generation length ranking prediction is enough. 

### Method

{{< justify >}}
We propose a simple but effective algorithm, for scheduling requests using ranking information, as detailed in Algorithm. The core idea is to iteratively run the predictor model $P$ to score new requests, then sort all requests according to their predicted generation length rankings. We form a running batch based on this sorted order, subject to memory or batch size constraints. To prevent the starvation of long requests, we've incorporated additional mechanisms, which we'll explain shortly. This ranking-based scheduling algorithm operates at the iteration level, making it compatible with established LLM serving techniques such as continuous batching and PagedAttention. 

{{< /justify >}}

### Training Length Ranking Predictor

We train the ranking predictor.

### Starvation Prevention

It is well known that simulating SJF will starve long requests. To mitigate this, we introduce starvation prevention.

##  Experiments

### Results
{{< justify >}}
Our proposed method improve the mean latency by up to 6.9x compared with FCFS and from 1.5x–1.9x compared with PO in Chatbot Serving. More detailed evaluations and comparisons can be found in our paper.
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
