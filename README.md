# Fine-Tuning LLMs Using Instructions and LoRA
Document my own journey in trying to understand the applications of LLMs in solving traditional NLP tasks, with a focus on instruction fine-tuning for classification.
 
Abstract: In a very short amount of time, large language models (LLMs) have seemingly dominated the NLP world with their impressive capabilities to generalise on unseen tasks. Two most popular techniques for LLMs are in-context learning (ICL) and chain-of-thought (CoT). However, crafting perfect  prompts is non-trivial: the choice, the number, and the order of the demonstrations can lead to dramatically different results, from near SOTA to near random guess. Instruction fine-tuning (IFT), on the other hand, refers to the process of training the models further on pairs of <instruction, output> to adapt the LLMs to our own tasks. Arguably, IFT gives rise to the emergent capabilities of LLMs by teaching them to follow instructions. In this presentation, I would like to argue that, for traditional NLP tasks such as classification, IFT is still a cheaper, more stable, and more efficient approach than ICL and CoT. The presentation will also cover tips on writing instructions and LoRA, one of the most popular parameter-efficient algorithms for fine-tuning LLMs while adapting only a small amount of the model’s parameters.

> [!NOTE]
> This does not mean that I'm denying the other aprroachs such as ICL or CoT, or the power of close sourced LLMs such as GPT-3.5/4. Of course if you don't have a coding background, then learning how to use GPT family is much easier than educating yourself on how to use, for example huggingface ecosystem. However, if you already know how to write models using transformers library, then, my main argument is that for traditional NLP tasks, fine-tuning is still cheaper, more efficient, and more stable than learning how to craft perfect prompts for your entire test set, especially when you have thousands of test samples.

# Table of Contents

1. [Why Fine-Tuning?](https://github.com/VanHoang85/my_LLMs_journey/tree/main#why-fine-tuning)
   * [What are ICL and CoT?](https://github.com/VanHoang85/my_LLMs_journey/tree/main#what-are-icl-and-cot)
   * [What are their disadvantages?](https://github.com/VanHoang85/my_LLMs_journey/tree/main#what-are-their-disadvantages)
2. [Instruction Fine-Tuning](https://github.com/VanHoang85/my_LLMs_journey/tree/main#instruction-fine-tuning-ift)
   * [What are instructions?](https://github.com/VanHoang85/my_LLMs_journey/tree/main#what-is-it)
   * [Guide on instruction writing](https://github.com/VanHoang85/my_LLMs_journey/tree/main#guide-on-instruction-writing)
   * [Do we really need instructions to fine-tune LLMs?](https://github.com/VanHoang85/my_LLMs_journey/tree/main#do-we-really-need-instructions-to-fine-tune-llms)
   * [Undesirable outputs](https://github.com/VanHoang85/my_LLMs_journey/tree/main#undesirable-outputs)
3. [Parameter-Efficient Fine-Tuning](https://github.com/VanHoang85/my_LLMs_journey/tree/main#parameter-efficient-fine-tuning-peft)
   * [Low-Rank Adaptation](https://github.com/VanHoang85/my_LLMs_journey/tree/main#lora-low-rank-adaptation)
4. [Loading and Training LLMs with Limited Resources](https://github.com/VanHoang85/my_LLMs_journey/tree/main#loading-and-training-llms-with-limited-resources)
5. [Additional Resources](https://github.com/VanHoang85/my_LLMs_journey/tree/main#additional-resources)

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

However, Wang et al. (2023)[^1] recently show that prompting the models with invalid reasoning steps can still achieve 80-90% performance of CoT using valid and sound reasoning. Additionally, ***best performing prompts often are semantically incoherent, task-irrelevant, unintuitive to humans, or even misleading*** (Prasad et al. 2023[^4]). If we put effort in collecting reasoning steps or crafting prompts, of course we want them to be useful. Yet, these findings raise the question of whether the models can really understand the reasoning and/or the demonstrations, and whether engineering human-friendly prompts is possible at all.

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
2. Additional details or constraints besides primary mentions of input and output, in general, do not improve model
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
* __Flan-T5 prediction:__ change
* __Llama 2 prediction:__ Based on the client's response, I would predict a Sustain attitude, indicating resistance against behaviour change. The client seems to downplay the amount of alcohol they consume, emphasizing that it's only for social gatherings and implying that they don't have a problem. This is a common way for individuals to resist changing their behaviours, as they may feel defensive or unwilling to acknowledge any potential issues. To further understand the client's attitudes and intentions, I would need to continue exploring this topic and assessing their level of motivation for change.

As illustrated above, Llama 2's default is to output answers with explanations as a conversation. It's nice if we want our customers to interact directly with the model. However, if our purpose is to use the LLMs as an annotation tool or as a backend component, the chatty feature makes it difficult for us to extract the exact labels in the answer. It's possible to use regular expressions to extract all mentions of the labels. However, we still don't know the correct label is the first or second or third mention. 

Therefore, I would highly recommend Flan-T5[^10] for classification tasks. The models are trained on 1k8+ NLP tasks and are shown to outperform other models with the same size up to 26% performance gain. There are 5 total models in the Flan-T5 family, ranging from 80M to 11B parameters[^14]. Additionally, Flan-T5's outputs are the labels themselves, unless you explicitly ask the models to generate reasoning. That means, if the models halluciate, you can simply replace the ill-formed outputs with a random label using regular expressions.

[^14]: https://huggingface.co/docs/transformers/model_doc/flan-t5

## Parameter-Efficient Fine-Tuning (PEFT)

As it is too costly to fine-tune the entire LLMs, PEFT resolves this issue by training only a small set of parameters, which might be (a) a subset of existing model parameters, or (b) a set of newly added parameters[^15]. As depicted in the figure below, this is a very active research area since it enables the adaptation of open-sourced LLMs to our own tasks in a more efficient manner. The survey by Lialin et al. (2023) covers 30 methods in total as of February 2023.

[Image Source: Lialin et al., 2023](https://arxiv.org/pdf/2303.15647.pdf)
<img width="600" alt="peft" src="https://github.com/VanHoang85/my_LLMs_journey/assets/38503004/2469d7a9-7e67-4b7b-9dad-fa3bb6b9a412">

Some common PEFT algorithms:
* Prefix-tuning (soft prompts / additive): train task specific embeddings and add to hidden states of all layers.
* BitFit (selective): train only the bias of the models.
* LoRA (reparametrization-based): decompose the weight changes into two low-rank matrices for training.

### LoRA (Low-Rank Adaptation)

Among all the PEFT methods, one of the most popular is LoRA by Hu et al. (2022)[^16]. The main intuition is that instead of training the full-rank weight update matrix, we can break it into two smaller rank matrices W<sub>A</sub> and W<sub>B</sub> for faster training. 

<img width="500" alt="image" src="https://github.com/VanHoang85/my_LLMs_journey/assets/38503004/844a4f77-0537-4133-ba6d-c59360ca7e1f">


In the normal training (on the left), we obtain the weight update (𝛿𝑊) by timing the learning rate with the negative gradient of the loss. Then we update the original weight matrix W and caculate the hidden state.

In LoRA, 𝛿𝑊 is broken down into W<sub>A</sub> and W<sub>B</sub> with a rank __r__ smaller than the dimension of both A and B. Then, we only need to do weight updates on these two matrices and compute the outputs for the hidden state separately. Sebastian Raschka (2023)[^17] wrote a great blog post on LoRA so interested readers are encouraged to go there for detailed explanation.

Hu et al. (2022)[^16] shows that LoRA outperforms full training (i.e., GPT-3 with 175B) while adapting only 2% of the parameters (i.e., 37.7M).

Some notes on LoRA hyper-parameters:
* **alpha 𝛼** is a scaling factor to adjust the LoRA weights, which defaults to 1.
* **rank r** controls the trade-off between model complexity and performance. Smaller rank leads to simpler matrices, meaning fewer parameters to learn, resulting faster training and less computational demands. LoRA's authors[^16] show in their paper that a rank of 1 is sufficient to obtain good performance. However, it might be the case that GPT-3 has already captured enough information for their test tasks. In other words, for your own tasks, you might need a higher rank to capture your task-specific information. 
* which **weight matrices** one wants to apply LoRA to. Each Transformer layer has 4 self-attention modules W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub>, W<sub>o</sub>, and two MLP. Which modules which layer to use is up to you. The original paper experiments mostly with W<sub>q</sub> and W<sub>k</sub> on the last layer of the models for simplicity and parameter-efficient. However, some has found that finet-tuning all layers with all modules, and/or longer epochs yields better performance[^18].

[^15]: Lialin et al. 2023. Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning. arXiv:2303.15647 [cs]. https://arxiv.org/pdf/2303.15647.pdf
[^16]: Hu et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR. https://arxiv.org/pdf/2106.09685.pdf
[^17]: Sebastian Raschka. 2023. Parameter-Efficient LLM Finetuning with Low-Rank Adaptation. [https://lightning.ai/pages/community/tutorial/lora-llm/](https://lightning.ai/pages/community/tutorial/lora-llm/)
[^18]: https://github.com/huggingface/peft/issues/622

***To merge or not to merge...***
After fine-tuning with LoRA, we can decide to keep the two matrices separately or we can merge them with the original matrix.

If we decide to keep them serpately, inference time increases due to additional computational steps. Using PEFT library on huggingface platform, similar to [this post](https://github.com/huggingface/peft/issues/217), I do feel inference time is like 10x slower, compared with ICL. After merging the LoRA weights and the original weights, inference time is fater but still ~2 times slower compared to ICL using original non-LoRA models. Hu et al. (2022)[^16] argues that the merged models incur no additional latency costs. Therefore, this issue might be due to implementation.

However, if we opt for merging, then instead of storing just the adapter weights, which is often very small (e.g., < 100MB), we have to store the full models (e.g., ~45GB for Flan-T5-XXL models). Image that you have 10 tasks, training and storing 10 LoRA adapters takes 100MB x 10 = ~1GB. On the other hands, merging helps with latency but you have to store 45GB x 10 = 450 GB.

Another caveat is that, even though PEFT[^15] is supposed to change only a subset of the model's weights. With prompt-tuning, you literally don't touch the model weights at all and only optimise the prompt embeddings. With BitFit, you change only the bias of the models. LoRA, however, means that you do change the entire model weights, just in a smart way. With that said, the LoRA-adapted models might have unexpected performance on unseen tasks. Imagine you have 10 tasks: ICL performance on 5 tasks is good enough but not the other 5. Therefore, you decide to fine-tune on the other 5 tasks and obtain improved performance as expected. Yet, what about the performance on the first 5 tasks now that you have overwritten the original weights? 

For this reason, it's more common to have a separate LoRA adapter for each task. Still, I think one of the advantages of IFT is its capabilities in a multi-task setting, and I do hope the research community will figure out new PEFT/LoRA methods for multi-task learning without incurring high costs.

## Loading and Training LLMs with limited resources

Question: Is it possible to fine-tune open-source LLMs on TUD's cluster?
Answer: Yes, yes, yes.

###Loading LLMs

On HuggingFace ecosystem, from Transformers 4.20.0, it supports [loading large models more efficiently](https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/model#large-model-loading).

```
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", load_in_8bit=True)
```

Argument information:
* **torch_dtype**: load the models in the desired `dtype`. With `auto`, it loads in the most optimal memory pattern.
* **low_cpu_mem_usage**: if `False`, we need twice the size of the model in RAM because it creates the full model, then loads the pretrained weights inside it. If `True`, we create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded.
* **device_map**: if `auto`, it optimises the loading of the models by splitting it across multiple devices. However, its default option means that it always tries to make use of all devices even though we can fit it in one GPU. As a result, it might throw an out-of-memory error. For example, ADAP-CLIN has 2 GPUs: Your model can fit into the other GPU but because of the activated `auto` argument, it tries to load your model in both GPUs but because one is being occupied, you get an error. The solution is to deactive this argument and use **low_cpu_mem_usage** to load on one GPU only.
* **load_in_8bit**: most models are trained in either 32 bit or 16 bit precision. However, one can quantize them into 8 bit (or 4 bit using **load_in_4bit** instead). Currently, you can only do inference when loaded in 4bit, not training. By rule of thumb, one needs half of the memory when loaded in 8 bit, and a quarter when loaded in 4 bit.
  * Flan-T5-XXL (11B) is 45 GB in size. Loaded in 8bit: 17 GB. Loaded in 4 bit: 13.5 GB.
  * Llama 2 (13B) is 27 GB in size. Loaded in 8bit: 13.5 GB. Loaded in 4 bit: 7 GB.
  * At TUD, we do have 48 GB GPUs.

Detailed guidelines about [how to load large models](https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/model#large-model-loading), [how to train LLMs on 1 GPU](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one) and [how to quantize LLMs models](https://huggingface.co/docs/transformers/main/en/main_classes/quantization).

###Training with PEFT

[PEFT library on huggingface](https://huggingface.co/docs/peft/index) is currently supporting 7 peft methods, including LoRA. To fine-tune the models with a supported method, in addition to loading and processing the dataset, loading the base model, we need to load the desired peft config and convert to peft model, which has been well supported by huggingface:

```
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# load the model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl",
                                                  load_in_8bit=True,
                                                  device_map="auto")
# define LoRA Config
config = LoraConfig(r=8,  # rank
                    lora_alpha=16,  # scaling factor
                    target_modules=["q", "v"],   # specify weight modules
                    lora_dropout=0.05,
                    inference_mode=False,
                    task_type=TaskType.SEQ_2_SEQ_LM)

# prepare model for training
model = prepare_model_for_kbit_training(model)

# add LoRA adaptor
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

By default, you apply LoRA on W_q and Q_v and on the last layer only. Currently, HF's peft doesn't give you an option to conveniently choose which modules or layers to apply LoRA on. To apply on all modules and layers, you can:

```
target_modules = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
config = LoraConfig(r=8,  # rank
                    lora_alpha=16,  # scaling factor
                    target_modules=target_modules
                    lora_dropout=0.05,
                    inference_mode=False,
                    task_type=TaskType.SEQ_2_SEQ_LM)
```

Sebastian Raschka wrote a blog post to document [his experiences after running thousands LoRA experiments](https://lightning.ai/pages/community/lora-insights/).

Detail examples about fine-tuning models on peft can be found [here](https://www.philschmid.de/fine-tune-flan-t5-peft#3-fine-tune-t5-with-lora-and-bnb-int-8) and [here](https://github.com/huggingface/peft/tree/main/examples).

## Additional Resources

For prompt/instruction writing:
* The course “ChatGPT Prompt Engineering for Developers” on [deepLearning.ai](https://www.deeplearning.ai/short-courses/).
* [OpenAI cookbook](https://github.com/openai/openai-cookbook/tree/main)
* [promptsource](https://github.com/bigscience-workshop/promptsource), which is a library to create and share promots among developers. It is compatible with huggingface ecosystem in which you can load a dataset from their database and then load all shared prompts for this dataset using `promptsource`.

Other PEFT platforms:
* [Adapter-Hub](https://github.com/Adapter-Hub/adapter-transformers)
* [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)

My personal recommendations about LLMs (in addition to the footnote/references):
* [The anti-hype LLM reading list](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)
* [Kaddour et al. 2023. Challenges and Applications of Large Language Models. arXiv:2307.10169 [cs]](https://arxiv.org/pdf/2307.10169.pdf): it requires some background knowledge on LLMs so it's more suitable for medium and advanced readers. 
* [Yang et al. 2023. Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond. arXiv:2304.13712 [cs].](https://arxiv.org/pdf/2304.13712.pdf): Provide a practical guide on whether you can take advantages of the LLMs' power or you'd better off sticking to the traditional DL models.
