# bafoGPT

🌐 [Homepage](#) | 🤗 [Hub](#) | 📖 [arXiv](#) | [GitHub](https://github.com/Motsepe-Jr/bafoGPT)

## Overview

Presenting BafoGT-2-2B-base and BafoGT-2-2B-it, open-source Zulu language models derived from the Gemma-2-2b architecture. These models underwent continuous pre-training on approximately 200 million tokens over 36 hours using 4 L40 GPUs. With a budget of under R10,000, they deliver performance comparable to models that typically require millions to train from scratch. 

Licensed under the permissive Gemma-2 and Apache 2.0 licenses, these models support both commercial use and further innovation. BafoGT-2-2B-base is designed for both IsiZulu and English languages, promoting research and innovation in African AI development. I hope this work inspires further contributions and advancements in this space.


## News

🔥 **[2024/08/20]:** The pretraining code has been released, and we also invite you to follow our Repo. 😆

## Main Contributions of this Project 

- **Vocab Expansion**: Expanded the Gemma-2-2B vocabulary size with 40,000 Zulu tokens for enhanced encoding/decoding efficiency.
- **Largest IsiZulu Dataset**: Open-sourced the largest IsiZulu supervised fine-tuning and pretraining datasets.
- **Large-Scale Data Collection and Processing Scripts**: Includes a separate repository for collecting, cleansing, and processing datasets.
- **Open Source**: Open sourcing both BafoGT-2-2B-base and BafoGT-2-2B-it, Our work is made possible by the open-source community. Special thanks to the teams behind [LitGPT](https://github.com/Lightning-AI/litgpt) and [Google's research](https://arxiv.org/pdf/2403.08295).

## Dataset

### Pretraining Dataset: 🤗 [Hub](https://huggingface.co/datasets/ChallengerSpaceShuttle/zulu-pretraining-dataset)

For the pretraining of **BafoGT**, I collected diverse internet-based data sources focused on both Zulu and English to facilitate bilingual understanding. The curriculum learning approach I employed was designed to help the model efficiently grasp the intricacies of Zulu by initially feeding it simpler datasets. The dataset was processed without shuffling, starting from simpler dictionary where zulu words are explained in English and progressing to more complex structures such as translated texts and transcriptions. 
 
- ** My hypothesis was that by starting continual pretraining with Zulu words defined in English, the base model would adjust its internal representations to map these Zulu words closer to their English counterparts, all while avoiding catastrophic forgetting.

The model first learn to map Zulu words and sentences to their English equivalents,  progressing to more complex data soruce such as wikipedia, new articles etc.

- **Books**: [Zulu-English Dictionary by Bishop Colenso](https://archive.org/details/zuluenglishdicti00brya) – A dictionary offering Zulu terms with English definitions, ideal for teaching basic word mappings.
- **Translation**: [South African Government Speeches](https://www.gov.za/news/inkulumo-echaza-isimo-sezwe-ithulwa-ngumhlonishwa-jg-zuma-umongameli-weriphabhulikhi) – Official speeches in Zulu, which help the model understand structured Zulu sentences and phrases.
- **Transcription**: [Zulu Community Corpus](https://corpora.uni-leipzig.de/en?corpusId=zul_community_2017) – A collection of transcriptions, exposing the model to real-life conversational Zulu.
- **Document**: [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en) – Text from various Zulu documents, helping the model understand written Zulu.
- **News**: [Isolezwe News](https://www.isolezwe.co.za/) – Zulu-language news articles for understanding contemporary Zulu language usage and current affairs.
- **Poems**: [Zulu Poetry](http://inkondlo_kazulu.co.za) – Poetry resources for cultural and literary language representation.
- **Web**: Various websites – Diverse web-scraped content in Zulu, providing a wide range of linguistic inputs.
- **Wikipedia**: [Zulu Wikipedia](https://zu.wikipedia.org/wiki/) – Articles in Zulu covering various topics, offering an informative and factual base for the model.

### Fine-tuning Dataset

For the fine-tuning phase, I leveraged large-scale datasets translated into Zulu. This included translating the entire **Cosmopedia** dataset from Hugging Face, along with a **WikiHow** dump translated into Zulu. The innovative aspect of this process was to train the model using bilingual question-answering pairs, where questions were posed in Zulu, and the model responded in English. This forced the model to not only understand Zulu but also to navigate between both languages effectively.

By training BafoGT this way, I encouraged cross-lingual knowledge transfer, enhancing the model’s capability to understand and generate coherent text in both Zulu and English. This fine-tuning method provided a robust base for both translation and generation tasks in these languages.

## Tokenizer

## Training

## Finetuning

## Model and Dataset Download

