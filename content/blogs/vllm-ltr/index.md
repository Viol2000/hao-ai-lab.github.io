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
      image = "img/llm-ltr.png"
      alt = "llm-ltr"
      caption = "An illustration of LLM-Ltr."
+++

{{< socialBadges arxiv-index="2408.15792" github="hao-ai-lab/vllm-ltr" >}}

**TL;DR:** Traditional Large Language Model (LLM) serving systems use first-come-first-serve scheduling since the exact output lengths are unpredictable. However, we developed a *learning to rank* approach that predicts the relative ranking of output lengths, enabling a more efficient scheduling policy (to approximate SJF) that reduced chatbot latency by 6.9x compared with FCFS.

## Background

### Head-of-line Blocking in LLM Serving 

LLMs are increasingly becoming the backbone of many Internet services and applications serving millions of users. In our setup, we serve an LLM (such as LLaMA3 70B) on a GPU cluster to process incoming user requests. Due to surging demand, efficient scheduling is crucial for maintaining high-quality service when many concurrent users compete for computing resources. For popular interactive applications like chatbots, this means minimizing user-perceived latency while maximizing system throughput to accommodate the highest possible number of users.

**Under high workload**, LLM services that implement a first-come-first-serve (FCFS) scheduling strategy inevitably face significant **Head-Of-Line (HOL) blocking**, as many requests must wait for others to execute. Figure 1 illustrates a typical example of how a long request can block shorter ones in FCFS scheduling, leading to significant HOL blocking. In such scenarios, it is well-established that the shortest-job-first (SJF) and shortest-remaining-time-first (SRTF) scheduling algorithms minimize the average latency experienced across all requests. However, SJF/SRTF are seldom implemented in LLM services because they require requests to be ordered by their remaining generation lengths, which is traditionally assumed to be difficult or impossible to know ahead of time in existing systems.

{{< image src="img/HOL.jpg" alt="HOL" width="120%" title="Figure 1: A long request can block short requests and introduce severe HOL blocking and high latency. We assume there is no prefill time, and the system takes 1 second to generate 1 token. With a First-come-first-serve (FCFS) schedule, the long request *R0*, which arrives first and takes 10 seconds to generate 10 tokens, will block subsequent shorter requests *R1* and *R2* for 10 seconds. Hence the latencies of *R0*,  *R1*, and *R2* are $10 / 10 = 1, (10 + 2) / 2 = 6, (10+2+1)/1=13 \mbox{ s / token}$, respectively, perceived by users, with an average latency of $(1+6+13)/3 = 6.67 \mbox{ s / token}$. By contrast, prioritizing shortest requests yields an average latency of $(1.3+1.5+1)/3=1.27 \mbox{ s / token}$ -- a $5.3\times$ reduction in average latency.">}}

### Learning to Rank

Learning to rank is a machine learning approach that learns to order items based on supervised ranking data. Among various ranking methods, we focus on listwise approaches that directly optimize the order of all items in a list. One such approach is (ListMLE)[https://dl.acm.org/doi/10.1145/1390156.1390306], a listwise ranking loss function of particular interest in our paper.

ListMLE minimizes the likelihood function defined as $\mathcal{\phi}(g(x),y)=-\log P\left(y \mid x ; g\right)$, where
$P(y \mid x ; g)=\prod_{i=1}^n \frac{\exp \left(g\left(x_{y(i)}\right)\right)}{\sum_{k=i}^n \exp \left(g\left(x_{y(k)}\right)\right)} $
Here, $P(y \mid x ; g)$ represents the probability of permutation $y$ given input $x$ and scoring function $g$. $x_{y(i)}$ denotes the element in $x$ corresponding to the $i$-th position in permutation $y$. The idea is to maximize the likelihood of the correct ranking $y$ by using scoring function $g$ to predict the ranking of input $x$. The loss function $\mathcal{\phi}(g(x),y)$ minimizes the negative log-likelihood of this probability, encouraging the model to predict rankings close to the true ordering.

## LLM Scheduling by Learning-To-Rank

### Ranking is All You Need (to appx. SJF)

We show that the precise generation length is not needed. An accurate generation length ranking prediction is enough. 

Ourgoal is to approximate true SJF/SRTF scheduling using these rankings to alleviate HOL blocking (Fig. 2 a) and obtain a relatively low latency in LLM serving. Ahigher Kendall’s Tau reflects a more accurate rank prediction against the oracle (i.e., SJF/SRTF), which empirically translates into higher end-to-end performance, as evidenced in Fig. 2 (b).
 
{{< image src="img/ranking.png" alt="ranking" width="120%" title="Figure2: (a): HOL blocking was evaluated by comparing FCFS and SRTF scheduling policies across 1K requests. (b): Analysis revealed that higher Kendall’s Tau correlation coefficients were associated with reduced latency. This finding was validated using the ShareGPT dataset with the Llama-3-8B model.">}}


### Method

{{< justify >}}
We propose a simple but effective algorithm, for scheduling requests using ranking information, as detailed in Algorithm. The core idea is to iteratively run the predictor model $P$ to score new requests, then sort all requests according to their predicted generation length rankings. We form a running batch based on this sorted order, subject to memory or batch size constraints. To prevent the starvation of long requests, we've incorporated additional mechanisms, which we'll explain shortly. This ranking-based scheduling algorithm operates at the iteration level, making it compatible with established LLM serving techniques such as continuous batching and PagedAttention. 

{{< /justify >}}


{{< image src="img/llm-ltr.png" alt="overview" width="80%" title="Overview of the method.">}}


### Training Length Ranking Predictor

For our predictor P, we utilize a small OPT model as the backbone, capable of processing natural language prompts as input and generating a score for ranking. While previous methods use classification (with bucketing) to generate accurate output length predictions, we find this approach both challenging and unnecessary. Instead, the relative ranking suffices. Based on this insight, we apply learning to rank to train the OPT model. This section explains how we train the OPT model as the predictor P to rank prompts by their expected generation length.

The model training process takes less than 5mins on 10K data, which can be obtained in real world serving. It benefits the real world serving.

### Starvation Prevention

It is well known that while SJF/SRTF scheduling can improve overall latency, it may lead to starvation for long requests, causing users to wait excessively for responses. Different from previous fairness-promoting design [39], which focuses on the fairness between different clients, we propose a *max_waiting_time* fairness metric to evaluate the fairness at per-request level (hence reflecting per-user satisfaction). We define max_waiting_time fairness by considering both Time To First Token (TTFT) and Time Per Output Token (TPOT)[12] in LLM serving as follows: 

 $max\_waiting\_time=max(TTFT,max(TPOT))$

Intuitively, $max\_waiting\_time$ characterizes the maximum time interval a user experiences between receiving two tokens after sending a request to the server. A larger max_waiting_time indicates a longer waiting time for the user to obtain a response, signifying more severe starvation. 

To mitigate starvation, our algorithm implements the following mechanism: 1) For each scheduling step, we increment a request’s starvation count if it is not executed. 2) When a request's starvation count reaches a pre-defined threshold, we will promote
 this request's priority by allocating "quantum" of execution time. 3) The request maintains this elevated
 priority until it exhausts its allocated quantum. This simple yet effective method
 prevents starvation at the request level, improves $max\_waiting\_time$, and ensures user satisfaction,
 as demonstrated in our experiments.

##  Experiments

### Main results

Fig. 3 compares the latency of our proposed ranking method with four baseline methods on ShareGPT
 and LMSYS-Chat-1M datasets with increasing arrival rates [2, 14, 12]. Under a rate of 64 requests/sec
ond, our method improves mean latency by up to 6.9x compared to FCFS and 1.5x–1.9x compared
 to PO. MLFQ and PO still face severe HOL blockings as they must run all requests for a certain time
 to obtain information for scheduling. PO must execute all arriving requests with the LLM to generate
 a length prediction. MLFQ must run all arriving requests before they enter the next priority level. The
 classification method optimizes for accuracy instead of ranking, missing optimization opportunities.
 While classification and our method still need to process all the requests first to obtain a prediction, using an OPT model takes less than 2% of the time (as shown in §5.5), thus greatly reducing HOL blocking.
 
{{< image src="img/main.png" alt="main.png" width="100%" title="Main results of LLM-Ltr">}}

### Overhead of the predictor

The predictor adds minimal overhead - less than 2% additional processing time across all settings. We use a 350M parameter predictor for the 70B model and a 125M predictor for the 8B model. While request processing involves both prefill and decode steps, the OPT predictor only performs prefill operations. The overhead increases with longer context lengths, which explains the higher overhead observed on the ShareGPT dataset.

{{< image src="img/overhead.png" alt="overhead" width="100%" title="Overhead of the predictor">}}

More detailed results can be found in the \S5 of our paper.

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
