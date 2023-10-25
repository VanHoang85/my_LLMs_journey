# Fine-Tuning LLMs Using Instructions and LoRa.
Document my own journey in trying to understand the applications of LLMs in solving traditional NLP tasks, with a focus on instruction fine-tuning for classification tasks.
 
Abstract: In a very short amount of time, large language models (LLMs) have seemingly dominated the NLP world with their impressive capabilities to generalise on unseen tasks. Two most popular techniques for LLMs are in-context learning (ICL) and chain-of-thought (CoT). However, crafting perfect  prompts is non-trivial: the choice, the number, and the order of the demonstrations can lead to dramatically different results, from near SOTA to near random guess. Instruction fine-tuning (IFT), on the other hand, refers to the process of training the models further on pairs of <instruction, output> to adapt the LLMs to our own tasks. Arguably, IFT gives rise to the emergent capabilities of LLMs by teaching them to follow instructions. In this presentation, I would like to argue that, for traditional NLP tasks such as classification, IFT is still a cheaper and more efficient approach than ICL and CoT. The presentation will also cover tips on writing instructions and LoRa, one of the most popular parameter-efficient algorithms for fine-tuning LLMs while adapting only a small amount of the model’s parameters.

> [!NOTE]
> This does not mean that I'm denying the other aprroachs such as ICL or CoT, or the power of close sourced LLMs such as GPT-3.5/4. Of course if you don't have a coding background, then learning how to use GPT family is much easier than educating yourself on how to use, for example huggingface ecosystem. However, if you already know how to write models using transformers library, then, my main argument is that for traditional NLP tasks, fine-tuning is still cheaper, more efficient, and more stable than learning how to craft perfect prompts for your entire test set, especially when you have thousands of test samples.


## Why Fine-Tuning?

There are two most popular padadigms to use with LLMs as they are without doing any parameter updates on the models' weights: In-Context Learning (ICL) and Chain-of-Thought (CoT).

**In-Context Learning** refers to the LLMs' abilities to learn to perform new tasks solely by "observing" the examples, or demonstrations, shown in the prompt. Strictly speaking, ICL doesn't include instructions, aka how the models should perform the task, in the prompt. If it does, then we call it **Instruction Following**. However, this distinction has become quite blur since prompts need both to yield good results on unseen tasks. Therefore, I will refer to ICL as if having instructions already. We call ICL _n_-shot if there are _n_ demonstrations in the prompt. Below is an example of ICL 2 shot:
```
Label the following review with either “positive” or “negative”.

Text: The price is a bit expensive, but the food is excellent!
Sentiment: positive.

Text: Food is good but the waiter is just rude to us!!!
Sentiment: negative

Text: Affordable price. Large portion.
Sentiment: 
```

**Chain-of_Thought** refers to the LLMs' abilities to perform tasks via a series of intermedia reasoning steps leading to the final coutcomes. To enable the reasoning capability, we add the phrase "Let’s think step by step." or "Give rationales before answering." into the prompt. An example is shown below (taken from Wang et al., 2023 [^1])
[^1] Wang et al. 2023. Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters. In ACL.

```
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
**Let’s think step by step.**

Answer: Originally, Leah had 32 chocolates and her sister had 42. So, in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.
```


## Instruction Fine-Tuning


## Parameter-Efficient Fine-Tuning (PEFT)


## Additional Resources


## Reference
