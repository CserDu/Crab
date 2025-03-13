<p align="center">
    <img src="assets/crab.jpeg" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/abs/2406.07476" style="color:#9C276A">
Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">


[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/ahsgdxhs/Crab) [![hf_dataset](https://img.shields.io/badge/🤗-MSVC-9C276A.svg)](https://huggingface.co/datasets/ahsgdxhs/AVUIE) [![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2406.07476) <br>

</h5>


## 📰 News


<img src="assets/teaser.png" width="800" />

## 🛠️ Requirements and Installation
Basic Dependencies:
* Python == 3.9
* Pytorch == 2.1.0
* transformers == 4.37.2
* deepspeed == 0.12.6

Install required packages:
```bash
git clone https://github.com/CserDu/Crab
cd Crab
pip install -r requirements.txt
```


## 🚀 Main Results

### Multi-Choice Video QA & Video Captioning
<p><img src="https://github.com/user-attachments/assets/e87fe4cf-07ea-4fde-998b-a0c63671c3b4" width="800" "/></p>

###  Open-Ended Video QA
<p><img src="https://github.com/user-attachments/assets/80b16c04-75ac-43b8-bc22-6952fdf994bb" width="800" "/></p>

### Audio QA 
<p><img src="https://github.com/user-attachments/assets/46e55952-5a54-4564-bcd4-cfa4edd7f36a" width="800" "/></p>

### Audio-Visual QA 
<p><img src="https://github.com/user-attachments/assets/8114c1e3-7f93-401b-9ea6-9ce7c96d7b05" width="800" "/></p>


## :earth_americas: Model Zoo
### Vision-only Checkpoints
| Model Name     | Model Type | Visual Encoder | Language Decoder | # Training Frames |
|:----------------|:------------:|:----------------|:------------------|:----------------:|
| [VideoLLaMA2-7B-Base](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B-Base)  | Base  | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  | 8 |
| [VideoLLaMA2-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B)  | Chat | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  | 8 |
| [VideoLLaMA2-7B-16F-Base](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base)  | Base  | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  | 16 |
| [VideoLLaMA2-7B-16F](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B-16F)  | Chat | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  | 16 |
| [VideoLLaMA2-8x7B-Base](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-8x7B-Base)  | Base | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)  | 8 |
| [VideoLLaMA2-8x7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-8x7B)  | Chat | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)  | 8 |
| [VideoLLaMA2-72B-Base](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-72B-Base)  | Base | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)  | 8 |
| [VideoLLaMA2-72B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-72B)  | Chat | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)  | 8 |
| [VideoLLaMA2.1-7B-16F-Base](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base) | Base | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)  | 16 |
| [VideoLLaMA2.1-7B-16F](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-16F)  | Chat | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)  | 16 |


### Audio-Visual Checkpoints
| Model Name     | Type | Audio Encoder | Language Decoder |
|:-------------------|:----------------|:----------------|:------------------|
| [VideoLLaMA2.1-7B-AV](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV)  | Chat | [Fine-tuned BEATs_iter3+(AS2M)(cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) | [VideoLLaMA2.1-7B-16F](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-16F)  |



## [🤗 Demo](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2)

It is highly recommended to try our [online demo](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2) first.

To run a video-based LLM (Large Language Model) web demonstration on your device, you will first need to ensure that you have the necessary model checkpoints prepared, followed by adhering to the steps outlined to successfully launch the demo.

### Single-model Version

* Launch a gradio app directly ([VideoLLaMA2-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B) is adopted by default):
```bash
python videollama2/serve/gradio_web_server_adhoc.py
```

### Multiple-model Version

1. Launch a global controller
```bash
cd /path/to/VideoLLaMA2
python -m videollama2.serve.controller --host 0.0.0.0 --port 10000
```

2. Launch a gradio webserver
```bash
python -m videollama2.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

3. Launch one or multiple model workers
```bash
#  export HF_ENDPOINT=https://hf-mirror.com  # If you are unable to access Hugging Face, try to uncomment this line.
python -m videollama2.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /PATH/TO/MODEL1
python -m videollama2.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model-path /PATH/TO/MODEL2
python -m videollama2.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40002 --worker http://localhost:40002 --model-path /PATH/TO/MODEL3
...
```


## 🗝️ Training & Evaluation

### Quick Start

To facilitate further development on top of our codebase, we provide a quick-start guide on how to train a customized [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) with [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) dataset and evaluate the trained model on the mainstream video-llm benchmarks.

1. Training Data Structure:
```bash
VideoLLaMA2
├── datasets
│   ├── videollava_pt
|   |   ├── llava_image/ # Available at: https://pan.baidu.com/s/17GYcE69FcJjjUM0e4Gad2w?pwd=9ga3 or https://drive.google.com/drive/folders/1QmFj2FcMAoWNCUyiUtdcW0-IOhLbOBcf?usp=drive_link
|   |   ├── valley/      # Available at: https://pan.baidu.com/s/1jluOimE7mmihEBfnpwwCew?pwd=jyjz or https://drive.google.com/drive/folders/1QmFj2FcMAoWNCUyiUtdcW0-IOhLbOBcf?usp=drive_link
|   |   └── valley_llavaimage.json # Available at: https://drive.google.com/file/d/1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD/view, including 703K video-text and 558K image-text pairs
│   ├── videollava_sft
|   |   ├── llava_image_tune/  # Available at: https://pan.baidu.com/s/1l-jT6t_DlN5DTklwArsqGw?pwd=o6ko
|   |   ├── videochatgpt_tune/ # Available at: https://pan.baidu.com/s/10hJ_U7wVmYTUo75YHc_n8g?pwd=g1hf
|   |   └── videochatgpt_llavaimage_tune.json # Available at: https://drive.google.com/file/d/1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD/view, including 100K video-centric, 625K image-centric and 40K text-only conversations
```
2. Command:
```bash
# VideoLLaMA2-vllava pretraining
bash scripts/vllava/pretrain.sh
# VideoLLaMA2-vllava finetuning
bash scripts/vllava/finetune.sh
```
3. Evaluation Data Structure:
```bash
VideoLLaMA2
├── eval
│   ├── egoschema # Official website: https://github.com/egoschema/EgoSchema
|   |   ├── good_clips_git/ # Available at: https://drive.google.com/drive/folders/1SS0VVz8rML1e5gWq7D7VtP1oxE2UtmhQ
|   |   └── questions.json  # Available at: https://github.com/egoschema/EgoSchema/blob/main/questions.json
│   ├── mvbench # Official website: https://huggingface.co/datasets/OpenGVLab/MVBench
|   |   ├── video/
|   |   |   ├── clever/
|   |   |   └── ...
|   |   └── json/
|   |   |   ├── action_antonym.json
|   |   |   └── ...
│   ├── perception_test_mcqa # Official website: https://huggingface.co/datasets/OpenGVLab/MVBench
|   |   ├── videos/ # Available at: https://storage.googleapis.com/dm-perception-test/zip_data/test_videos.zip
|   |   └── mc_question_test.json # Download from https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_test_annotations.zip
│   ├── videomme # Official website: https://video-mme.github.io/home_page.html#leaderboard
|   |   ├── test-00000-of-00001.parquet
|   |   ├── videos/
|   |   └── subtitles/
│   ├── Activitynet_Zero_Shot_QA # Official website: https://github.com/MILVLG/activitynet-qa
|   |   ├── all_test/   # Available at: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW
|   |   ├── test_q.json # Available at: https://github.com/MILVLG/activitynet-qa/tree/master/dataset
|   |   └── test_a.json # Available at: https://github.com/MILVLG/activitynet-qa/tree/master/dataset
│   ├── MSVD_Zero_Shot_QA # Official website: https://github.com/xudejing/video-question-answering
|   |   ├── videos/     
|   |   ├── test_q.json 
|   |   └── test_a.json
│   ├── videochatgpt_gen # Official website: https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/quantitative_evaluation
|   |   ├── Test_Videos/ # Available at: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW
|   |   ├── Test_Human_Annotated_Captions/ # Available at: https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FQuantitative%5FEvaluation%2Fbenchamarking%2FTest%5FHuman%5FAnnotated%5FCaptions%2Ezip&parent=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FQuantitative%5FEvaluation%2Fbenchamarking&ga=1
|   |   ├── generic_qa.json     # These three json files available at: https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FQuantitative%5FEvaluation%2Fbenchamarking%2FBenchmarking%5FQA&ga=1
|   |   ├── temporal_qa.json
|   |   └── consistency_qa.json
```
4. Command:
```bash
# mvbench evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_video_qa_mvbench.sh
# activitynet-qa evaluation (need to set azure openai key/endpoint/deployname)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_video_qa_mvbench.sh
```

### Data Format

If you want to train a video-llm on your data, you need to follow the procedures below to prepare the video/image sft data:

1. Suppose your data structure is like:
```bash
VideoLLaMA2
├── datasets
│   ├── custom_sft
│   |   ├── images
│   |   ├── videos
|   |   └── custom.json
```
2. Then you should re-organize the annotated video/image sft data according to the following format:
```json
[
    {
        "id": 0,
        "video": "images/xxx.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            ...
        ],
    }
    {
        "id": 1,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ],
    },
    ...
]
```
3. Modify the `scripts/custom/finetune.sh`:
```bash
...
--data_path datasets/custom_sft/custom.json
--data_folder datasets/custom_sft/
--pretrain_mm_mlp_adapter CONNECTOR_DOWNLOAD_PATH (e.g., DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base)
...
```

## 🤖 Inference

Video/Image Inference:
```python
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


def inference():
    disable_torch_init()

    # Video Inference
    modal = 'video'
    modal_path = 'assets/cat_and_chicken.mp4' 
    instruct = 'What animals are in the video, what are they doing, and how does the video feel?'
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.

    # Image Inference
    modal = 'image'
    modal_path = 'assets/sora.png'
    instruct = 'What is the woman wearing, what is she doing, and how does the image feel?'
    # Reply:
    # The woman in the image is wearing a black coat and sunglasses, and she is walking down a rain-soaked city street. The image feels vibrant and lively, with the bright city lights reflecting off the wet pavement, creating a visually appealing atmosphere. The woman's presence adds a sense of style and confidence to the scene, as she navigates the bustling urban environment.

    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base'
    model, processor, tokenizer = model_init(model_path)
    output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    print(output)

if __name__ == "__main__":
    inference()
```

## 📑 Citation

If you find VideoLLaMA useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{damonlpsg2024videollama2,
  title={VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs},
  author={Cheng, Zesen and Leng, Sicong and Zhang, Hang and Xin, Yifei and Li, Xin and Chen, Guanzheng and Zhu, Yongxin and Zhang, Wenqi and Luo, Ziyang and Zhao, Deli and Bing, Lidong},
  journal={arXiv preprint arXiv:2406.07476},
  year={2024},
  url = {https://arxiv.org/abs/2406.07476}
}

@article{damonlpsg2023videollama,
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  author = {Zhang, Hang and Li, Xin and Bing, Lidong},
  journal = {arXiv preprint arXiv:2306.02858},
  year = {2023},
  url = {https://arxiv.org/abs/2306.02858}
}
```

## 👍 Acknowledgement
The codebase of VideoLLaMA 2 is adapted from [**LLaVA 1.5**](https:github.com/haotian-liu/LLaVA) and [**FastChat**](https://github.com/lm-sys/FastChat). We are also grateful for the following projects our VideoLLaMA 2 arise from:
* [**LLaMA 2**](https://github.com/meta-llama/llama), [**Mistral-7B**](https://mistral.ai/news/announcing-mistral-7b/), [**OpenAI CLIP**](https://openai.com/index/clip/), [**Qwen2**](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f), [**SigLIP**](https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba), [**Honeybee**](https://github.com/kakaobrain/honeybee).
* [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT), [**Video-LLaVA**](https://github.com/PKU-YuanGroup/Video-LLaVA). 
* [**WebVid**](https://github.com/m-bain/webvid), [**Panda-70M**](https://github.com/snap-research/Panda-70M), [**LanguageBind**](https://github.com/PKU-YuanGroup/LanguageBind), [**InternVid**](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid).
* [**VideoChat2**](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [**Valley**](https://github.com/RupertLuo/Valley), [**VTimeLLM**](https://github.com/huangb23/VTimeLLM), [**ShareGPT4V**](https://sharegpt4v.github.io/).
* [**Magpie**](https://github.com/magpie-align/magpie), [**ALLaVA**](https://github.com/FreedomIntelligence/ALLaVA), [**AVInstruct**](https://github.com/rikeilong/Bay-CAT/tree/main/AVinstruct).


## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**, subject to the model Licenses of LLaMA and Mistral, Terms of Use of the data generated by OpenAI, and Privacy Practices of ShareGPT. Please get in touch with us if you find any potential violations.
