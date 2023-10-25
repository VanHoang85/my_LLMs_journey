# Fine-Tuning LLMs Using Instructions and LoRa.
Document my own journey in trying to understand the applications of LLMs in solving traditional NLP tasks, with a focus on instruction fine-tuning for classification tasks.
 
Abstract: In a very short amount of time, large language models (LLMs) have seemingly dominated the NLP world with their impressive capabilities to generalise on unseen tasks. Two most popular techniques for LLMs are in-context learning (ICL) and chain-of-thought (CoT). However, crafting perfect  prompts is non-trivial: the choice, the number, and the order of the demonstrations can lead to dramatically different results, from near SOTA to near random guess. Instruction fine-tuning (IFT), on the other hand, refers to the process of training the models further on pairs of <instruction, output> to adapt the LLMs to our own tasks. Arguably, IFT gives rise to the emergent capabilities of LLMs by teaching them to follow instructions. In this presentation, I would like to argue that, for traditional NLP tasks such as classification, IFT is still a cheaper and more efficient approach than ICL and CoT. The presentation will also cover tips on writing instructions and LoRa, one of the most popular parameter-efficient algorithms for fine-tuning LLMs while adapting only a small amount of the model’s parameters.

> [!NOTE]
> This does not mean that I'm denying the other aprroachs such as ICL or CoT, or the power of close sourced LLMs such as GPT-3.5/4. Of course if you don't have a coding background, then learning how to use GPT family is much easier than educating yourself on how to use, for example huggingface ecosystem. However, if you already know how to write models using transformers library, then, my main argument is that for traditional NLP tasks, fine-tuning is still cheaper, more efficient, and more stable than learning how to craft perfect prompts for your entire test set, especially when you have thousands of test samples.


## Why Fine-Tuning?

### What are ICL and CoT?
There are two most popular padadigms to use with LLMs as they are without doing any parameter updates on the models' weights: In-Context Learning (ICL) and Chain-of-Thought (CoT).

***In-Context Learning*** refers to the LLMs' abilities to learn to perform new tasks solely by "observing" the examples, or demonstrations, shown in the prompt. Strictly speaking, ICL doesn't include instructions, aka how the models should perform the task, in the prompt. If it does, then we call it **Instruction Following**. However, this distinction has become quite blur since prompts need both to yield good results on unseen tasks. Therefore, I will refer to ICL as if having instructions already. We call ICL _n_-shot if there are _n_ demonstrations in the prompt. Below is an example of ICL 2 shot:
```
Label the following review with either “positive” or “negative”.

Example 1:
Text: The price is a bit expensive, but the food is excellent!
Sentiment: positive.

Example 2:
Text: Food is good but the waiter is just rude to us!!!
Sentiment: negative

Text: Affordable price. Large portion.
Sentiment: 
```

***Chain-of-Thought*** refers to the LLMs' abilities to perform tasks via a series of intermedia reasoning steps leading to the final coutcomes. To enable the reasoning capability, we add the phrase "Let’s think step by step." or "Give rationales before answering." into the prompt. An example is shown below (taken from Wang et al., 2023[^1])
[^1]: Wang et al. 2023. Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters. In ACL. https://aclanthology.org/2023.acl-long.153/

```
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let’s think step by step.

Answer: Originally, Leah had 32 chocolates and her sister had 42. So, in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.
```

### What are their disadvantages?

Crafting perfect prompts is a ***non-trivial*** task. If you only have a few test cases, then you can quickly assess the prompts' performance. However, when dealing with traditional NLP tasks, we are more likely to deal with thousand test samples. My own experience when using GPT-3.5 is that just adding or removing the formatting request (e.g., "Format the output as a dictionary with 'Answer' and 'Reasoning' as keys") can change the final answer when it shouldn't be the case.

More importantly, for ICL, the prompt format, the number, the choice, and the order of the demonstrations can lead to different results, from near SOTA to near random guess (Zhao et al. 2021[^2]). The authors discover several biases of GPT-3 when performing classification tasks to explain this brittleness phenomenon of prompt engeering. One of the bias is the majority label bias, meaning that GPT-3 simply repeats the only label of the demonstration in ICL 1-shot, which exaplains a drop in performance when moving from ICL 0-shot to 1-shot. However, when playing around with GPT-3.5 for my own classification task, I didn't observe this bias at all: out of 600 samples, GPT-3.5 only repeats the label in ~60 times. Such difference might be due to an upgrade from GPT-3 to GPT-3.5. Adding more demonstrations, unfortunately, does not necessarily lead to better results.

[^2]: Zhao et al. 2021. Calibrate Before Use: Improving Few-shot Performance of Language Models. In ICML. https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf

For CoT, it mostly benefits complicated reasoning tasks such as math and with large models (e.g. 50B+ parameters) while simple tasks only benefit slightly from CoT prompting[^3]. The question is whether CoT helps with traditional NLP tasks. My observation is that when dealing with tasks that need expert knowledge (e.g., psychotherapy), unless one can check the rationales themselves, it will take a huge amount of time, effort, and costs to hire experts for verification purposes. However, Wang et al. (2023)[^1] recently show that prompting the models with invalid reasoning steps can still achieve 80-90% performance of CoT using valid and sound reasoning. 

Additionally, best performing prompts often are semantically incoherent, task-irrelevant, unintuitive to humans, or even misleading (Prasad et al. 2023[^4]). If we put effort in collecting reasoning steps or crafting prompts, of course we want them to be useful. Yet, these findings raise the question of whether the models can really understand the reasoning and the demonstrations at all, and whether it is worths our efforts after all.

[^3]: Lilian Weng. 2023. Prompt Engineering. https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
[^4]: Prasad et al. 2023. GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models. In EACL. https://aclanthology.org/2023.eacl-main.277/

Due to these difficulties, one will eventually move from hard prompts (i.e., manually written by humans) to soft prompts[^5] (i.e., learnable tensors concatenated with the input embedding that can be optimised to tasks). Many algorithms employ a model for prompt optimisation. Consequently, it's just matter of which part one wants to fine-tune: the prompt, or the model weights.

[^5]: https://huggingface.co/docs/peft/conceptual_guides/prompting

## Instruction Fine-Tuning

<img width="776" alt="vanilla FT versus instruction FT" src="https://github.com/VanHoang85/my_LLMs_journey/assets/38503004/c53c5d4d-88ca-4422-b8d7-f87efa93ac8d">


## Parameter-Efficient Fine-Tuning (PEFT)


## Additional Resources


## References
