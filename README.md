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

More importantly, for ICL, ***the prompt format, the number, the choice, and the order of the demonstrations can lead to different results, from near SOTA to near random guess*** (Zhao et al. 2021[^2]). The authors discover several biases of GPT-3 when performing classification tasks to explain this brittleness phenomenon of prompt engeering. One of the bias is the majority label bias, meaning that GPT-3 simply repeats the only label of the demonstration in ICL 1-shot, which exaplains a drop in performance when moving from ICL 0-shot to 1-shot. However, when playing around with GPT-3.5 for my own classification task, I didn't observe this bias at all: out of 600 samples, GPT-3.5 only repeats the label in ~60 times. Such difference might be due to an upgrade from GPT-3 to GPT-3.5. Adding more demonstrations, unfortunately, does not necessarily lead to better results.

[^2]: Zhao et al. 2021. Calibrate Before Use: Improving Few-shot Performance of Language Models. In ICML. https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf

For CoT, it ***mostly benefits complicated reasoning tasks such as math and with large models*** (e.g. 50B+ parameters) while simple tasks only benefit slightly from CoT prompting[^3]. The question is whether CoT helps with traditional NLP tasks. My observation is that when dealing with tasks that need expert knowledge (e.g., psychotherapy), unless one can check the rationales themselves, it will take a huge amount of time, effort, and costs to hire experts for verification purposes. 

However, Wang et al. (2023)[^1] recently show that prompting the models with invalid reasoning steps can still achieve 80-90% performance of CoT using valid and sound reasoning. Additionally, ***best performing prompts often are semantically incoherent, task-irrelevant, unintuitive to humans, or even misleading*** (Prasad et al. 2023[^4]). If we put effort in collecting reasoning steps or crafting prompts, of course we want them to be useful. Yet, these findings raise the question of whether the models can really understand the reasoning and the demonstrations at all, and whether it is worth our efforts after all.

[^3]: Lilian Weng. 2023. Prompt Engineering. https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
[^4]: Prasad et al. 2023. GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models. In EACL. https://aclanthology.org/2023.eacl-main.277/

Due to these difficulties, one will eventually move from hard prompts (i.e., manually written by humans) to soft prompts[^5] (i.e., learnable tensors concatenated with the input embedding that can be optimised to tasks). Many algorithms employ a model for prompt optimisation. Consequently, it's just matter of which part one wants to fine-tune: the prompt, or the model weights.

[^5]: https://huggingface.co/docs/peft/conceptual_guides/prompting

## Instruction Fine-Tuning (IFT)

> [!NOTE]
> Strictly speaking, instructions refer to the guidances on what the models should perform while prompts are generally about the entire text, including test sample, demonstrations (if any), additional requirements (if any), restrictions (if any). However, literature on IFT often consider instructions as the entire text as inputs into the models for training. Therefore, I will use them interchangeably in my writing though I tend to use "instructions" with IFT and "prompts" for ICL/CoT.

### What is it?

<img width="600" alt="vanilla FT versus instruction FT" src="https://github.com/VanHoang85/my_LLMs_journey/assets/38503004/c53c5d4d-88ca-4422-b8d7-f87efa93ac8d">

As depicted in the above figure on the left, vanilla FT for classification tasks on BERT/LSTM/RNN models requires having data points as pairs of <input, output>. After encoding the entire input sequence, BERT makes use of the first token of the last hidden state, aka the classification token <cls>, putting it through a softmax layer to obtain the prediction. The predictions are of numeric value and we thus need to map it to a label (e.g., positive or negative or neutral).

When using LLMs, all tasks are framed as generation problem, including classification. That means, the LLMs have to generate exactly the label of the test sample, which is "positive", as illustrated in the figure above. Another change is that we fine-tune the LLMs using a dataset consisting of pairs of <instruction, output> and the input is included in the instruction. Arguably, IFT is what give rises to the emergent capabilities of LLMs by teaching them to follow instructions and generalise on unseen tasks[^6]. 

[^6]: https://github.com/xiaoya-li/Instruction-Tuning-Survey

### Guide on instruction writing

A task's instruction can be formulatted in different ways. For sentiment analysis, some examples are shown below, in which besides the wording of task, one can (1) specify the output space as a list of labels, (2) elaborate the meaning with information, and/or (3) make it like a cloze test. 

```
1. Predict the sentiment of the following text: <input_text>. Options are “positive”, “negative”, or “neutral”.

2. What is the sentiment of the review? Choose “positive” if the customer likes the place, “negative” if the customer hates it, and “neutral” if there is no information. Review: <input_text>. Answer:

3. Review: <input_text>. Select the correct sentiment of the review: (a) positive (good responses from customers), (b) negative (bad responses from customers), (c) neutral (no information available).
```

Does it make any difference? Motivated by a similar question, Yin et al. (2023)[^7] conduct a systematic study to understand the role of task definitions in instruction learning. The authors define 8 categories of task definitions (as illustrated below), and manually annotate 100 samples of 757 training tasks and 119 unseen test tasks in the English portion of Natural Instruction dataset[^8]. Then, the models are re-trained with each category ablated out and the performance is measured on validation set with the same ablation.

[^7]: Yin et al. 2023. Did You Read the Instructions? Rethinking the Effectiveness of Task Definitions in Instruction Learning. In ACL. https://aclanthology.org/2023.acl-long.172.pdf
[^8]: Wang et al. 2022. Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. In EMNLP. https://aclanthology.org/2022.emnlp-main.340/

<img width="600" alt="role of task instructions" src="https://github.com/VanHoang85/my_LLMs_journey/assets/38503004/f0f74e51-6a72-4a03-92b6-743f628c4fd5">

The answers to their RQ1 (i.e., Which parts of task definitions are important when performing zero-shot instruction learning?) are as follow. Readers who are intereted in understanding instruction/prompt writing process are encouraged to read the entire paper. I myself enjoy it :).

```
1. For classification, label-related information is the most crucial since it helps identify the output space and each label's meaning when generalising to unseen tasks.
2.  Additional details or constraints besides primary mentions of input and output, in general, do not improve model
performance. As model size increases, additional details become important.
3. Task definitions can be extensively compressed with no performance degradation, particularly for generation tasks.
```

### Do we really need instructions to fine-tune LLMs?

Learning how to write instructions can sound overwhelming at first. And you might wonder whether it is possible to fine-tune the LLMs without instructions at all. The answer is yes; one can fine-tune LLMs using pairs of <input, output> similar to vanilla FT. However, there are advantages in using instructions, and it's not just because the skill of writing instructions can translate into better prompts for ICL.

Gupta et al. (2023)[^9] show that ***IFT enable learning with limited data***:
* In single-task setting, their instruction-tuned models only need 25% of downstream training data (~1k samples) to outperform the SOTA but non-instruction-tuned models trained with 100% data.
* In multi-task setting, they only need 6% of data (~200 samples) for each task to get comparable results to the SOTA but non-instruction-tuned models trained with 100% data.

[^9]: Gupta et al. 2023. Instruction Tuned Models are Quick Learners. arXiv:2306.05539 [cs]. https://arxiv.org/pdf/2306.05539.pdf

Longpre et al. (2023) shows that ***IFT enhances single-task FT***:
* When employing instruction-tuned Flan-T5 models as the starting checkpoint, the models converge more quickly and yield better results compared to non-instruction-tuned T5 models.
  
[^10]: Longpre et al. 2023. The Flan Collection: Designing Data and Methods for Effective Instruction Tuning. In ICML. https://proceedings.mlr.press/v202/longpre23a/longpre23a.pdf

My personal experiences:
* IFT does works with limited data. For my own tasks, IFT with 100-200 samples is already enough to either outperform or get comparable performance to GPT-3.5 using ICL. Furthermore, IFT often yields better results with more data while ICL performance is stable regardless of the number of demonstrations it can choose from.
* Interested readers can check out on RAFT (Real-world Annotated Few-shot Tasks) leaderboard[^11], Schick et al. 2022[^12], and Liu et al. 2022[^13] in which the papers demonstrate IFT's learning capabilities with 50 samples, outperforming GPT-3 with ICL 0-shot.
*  When I fine-tune on my own dataset of 4k samples using instruction-tuned Flan-T5-XXL (11B), it takes ~22 hours for the model to converge with vanilla FT while IFT only takes ~7 hours. Despite similar results on validation set, the difference in training time is enough to convince me to use instructions.
*  You do get better at writing instructions and/or prompts :) Practice makes perfect.

[^11]: https://huggingface.co/spaces/ought/raft-leaderboard
[^12]: Schick et al. 2022. True Few-Shot Learning with Prompts—A Real-World Perspective. In Transactions of the Association for Computational Linguistics. https://aclanthology.org/2022.tacl-1.41/
[^13]: Liu et al. 2022. Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. In NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2022/file/0cde695b83bd186c1fd456302888454c-Paper-Conference.pdf

### Undesirable Outputs

As stated above, all tasks are framed as generation problem. Therefore, the models might generate ill-formed outputs. In my tasks, instead of generating class label, the models produce texts such as `I want to go I want to go` when fine-tuning with fewer than 100 samples.

Moreover, a tip when choosing your base model is to pay attention to its default output formatting. For example:
* __Given an instruction:__ Label the client utterance as either “change”, “neutral”, or “sustain”. Client: “I-I don't think I'm drinking that much. I mean, it's-it's mainly for social gatherings. Like it's nothing that I do, like by myself or whatever.” Answer:
* __Flan-T5 prediction:__: change
* __Llama 2 prediction:__: Based on the client's response, I would predict a Sustain attitude, indicating resistance against behaviour change. The client seems to downplay the amount of alcohol they consume, emphasizing that it's only for social gatherings and implying that they don't have a problem. This is a common way for individuals to resist changing their behaviours, as they may feel defensive or unwilling to acknowledge any potential issues. To further understand the client's attitudes and intentions, I would need to continue exploring this topic and assessing their level of motivation for change.

```
```

As illustrated above, Llama 2's default is to output answers with explanations as a conversation. It's nice if we want our customers to interact directly with the model. However, if our purpose is to use the LLMs as an annotation tool or as a backend component, the chatty feature makes it difficult for us to extract the exact labels in the answer. It's possible to use regular expressions to extract all mentions of the labels. However, we still don't know the correct label is the first or second or third mention. 

Therefore, I would highly recommend Flan-T5[^10] for classification tasks. The models are trained on 1k8+ NLP tasks and are shown to outperform other models with the same size up to 26% performance gain. There are 5 total models in the Flan-T5 family, ranging from 80M to 11B parameters[^14]. Additionally, Flan-T5's outputs are the labels themselves, unless you explicitly ask the models to generate reasoning. That means, if the models halluciate, you can simply replace the ill-formed outputs with a random label using regular expressions.

[^14]: https://huggingface.co/docs/transformers/model_doc/flan-t5

## Parameter-Efficient Fine-Tuning (PEFT)


## Additional Resources


## References
