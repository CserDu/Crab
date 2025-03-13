<p align="center">
    <img src="assets/crab.jpeg" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/abs/2406.07476" style="color:#9C276A">
Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">


[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/ahsgdxhs/Crab) [![hf_data](https://img.shields.io/badge/🤗-MSVC-9C276A.svg)](https://huggingface.co/datasets/ahsgdxhs/AVUIE) [![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2406.07476) <br>

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


## 🚀 Quick Start
1. Download [finetune weights](https://huggingface.co/ahsgdxhs/Crab)
2. Command:
```bash
bash scripts/finetune/inference_hyper_lora.sh
``` 


## 🗝️ Training
1. Downlod [AVUIE dataset annotations](https://huggingface.co/datasets/ahsgdxhs/AVUIE) and raw videos from [AVE](https://github.com/YapengTian/AVE-ECCV18), [AVVP](https://github.com/YapengTian/AVVP-ECCV20), [AVS](https://github.com/OpenNLPLab/AVSBench), [Ref-AVS](https://github.com/GeWu-Lab/Ref-AVS), [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA), [VALOR](https://github.com/TXH-mercury/VALOR)
2. command:
```bash
bash scripts/finetune/finetun_hyper_lora.sh
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
