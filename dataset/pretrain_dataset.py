import json
import ast
import os
import csv
import librosa
from os.path import join
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
import torchaudio.compliance.kaldi as ta_kaldi

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from decord import VideoReader
from transformers import CLIPImageProcessor

from dataset.audio_processor import preprocess

'''
Image caption
Video caption
Audio caption
Grounded VQA
'''
class PretrainDataset(Dataset):

    def __init__(
        self,
        image_annotation_path='/group/40061/cserdu/data/video-llava/train_json/llava_image_.json',
        video_annotation_path='/group/40061/cserdu/data/video-llava/train_json/valid_valley_.json',
        video_llava_data_root='/group/40061/cserdu/data/video-llava',
        image_caption_task=False,
        video_caption_task=False,
        image_size = 224,
        video_frame_nums = 8,
        # wavcaps_data_root='/DATA/DATANAS2/ligy/WavCaps',
        audiocaps_data_root='/group/40061/cserdu/data/AudioCaps',
        audio_caption_task=False,
        grounded_vqa_data_root='/group/40061/cserdu/data/GroundedVQA',
        segmentation_task=False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        lvis_data_root='/group/40061/cserdu/data/LVIS',
        video_processor: CLIPImageProcessor = None,
        # audio_processor=None,
        tokenizer: transformers.PreTrainedTokenizer = None,
    ) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        token_nums = image_scale_nums * token_nums_per_scale
        mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        self.mask_token = ''.join(mask_token)
        print('mask token: ',self.mask_token)

        self.samples = []

        if image_caption_task:
            self.add_image_caption_samples(image_annotation_path,video_llava_data_root,max_sample_nums=None)
        
        if video_caption_task:
            self.add_video_caption_samples(video_annotation_path,video_llava_data_root,max_sample_nums=None)
    
        if audio_caption_task:
            self.add_audio_caption_samples(audiocaps_data_root,max_sample_nums=None)

        if segmentation_task:
            # self.add_grounded_vqa_samples(grounded_vqa_data_root,max_sample_nums=None)
            self.add_lvis_segmentation_samples(lvis_data_root,max_sample_nums=None)

        self.video_processor = video_processor
        self.tokenizer = tokenizer
        

    def add_image_caption_samples(self,image_annotation_path,video_llava_data_root,max_sample_nums=None):
        tot = 0
        with open(image_annotation_path,'r') as f:
            samples = json.load(f)
            for sample in samples:
                image = sample['image']
                image_path = join(video_llava_data_root,image)
                conversations = sample['conversations']
                instruction = conversations[0]['value']
                question = instruction.replace('<image>','')
                question = question.replace('\n','')
                instruction = f'This is an image:\n<image_start><image><image_end>\nPlease answer the question:\n{question}'
                output = conversations[1]['value']
                if output[-1] not in ['.','!','?']:
                    output += '.'
                self.samples.append(
                    {
                        'image':image_path,
                        'instruction':instruction,
                        'output':output,
                        'question':question
                    }
                )
                tot+=1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'image caption sample nums: {tot}')

    
    def add_video_caption_samples(self,video_annotation_path,video_llava_data_root,max_sample_nums=None):
        tot = 0
        with open(video_annotation_path,'r') as f:
            samples = json.load(f)
            for sample in samples:
                video = sample['video']
                video_path = join(video_llava_data_root,video)
                conversations = sample['conversations']
                instruction = conversations[0]['value']
                question = instruction.replace('<video>','')
                question = question.replace('\n','')
                instruction = f'This is a video:\n<video_start><video><video_end>\nPlease answer the question:\n{question}'
                output = conversations[1]['value']
                if output[-1] not in ['.','!','?']:
                    output += '.'
                self.samples.append(
                    {
                        'video':video_path,
                        'instruction':instruction,
                        'output':output,
                        'question':question
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'video caption sample nums: {tot}')


    def add_audio_caption_samples(self,audiocaps_data_root,max_sample_nums=None):
        '''AudioCaps data'''
        tot = 0
        with open(join(audiocaps_data_root,'train.json'),'r') as f:
            samples = json.load(f)
            for i,sample in enumerate(samples):
                audiocap_id = sample['audiocap_id']
                if audiocap_id == '12347':
                    continue
                start_time = sample['start_time']
                caption = sample['caption']
                audio_path = join(audiocaps_data_root,'data',f'{audiocap_id}.wav')
                self.samples.append(
                    {
                        'audio':audio_path,
                        'instruction':'This is an audio:\n<audio_start><audio><audio_end>\nPlease describe this audio.',
                        'output':caption,
                        'question':'Please describe this audio.',
                    }
                )
                tot += 1 
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        with open(join(audiocaps_data_root,'val.json'),'r') as f:
            samples = json.load(f)
            for i,sample in enumerate(samples):
                audiocap_id = sample['audiocap_id']
                start_time = sample['start_time']
                caption = sample['caption']
                audio_path = join(audiocaps_data_root,'data',f'{audiocap_id}.wav')
                self.samples.append(
                    {
                        'audio':audio_path,
                        'instruction':'This is an audio:\n<audio_start><audio><audio_end>\nPlease describe this audio.',
                        'output':caption,
                        'question':'Please describe this audio.',
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'AudioCaps sample nums: {tot}')


    def add_grounded_vqa_samples(self,grounded_vqa_data_root,max_sample_nums=None):
        tot=0
        with open(join(grounded_vqa_data_root,'train_grounding.json'),'r') as f:
            samples = json.load(f)
            for vname,value in samples.items():
                question = value['question']
                if question[-1] != '?':
                    question = question + '?'
                most_common_answer = value['most_common_answer']
                image_path = join(grounded_vqa_data_root,'train',vname)
                mask_path = join(grounded_vqa_data_root,'train',vname[:-4]+'.png')
                self.samples.append(
                    {
                        'image':image_path,
                        'instruction':f'<image_start><image><image_end>\n{question}\nPlease answer this question and segment the corresponding area in the image.',
                        'output':f'The answer to the question is {most_common_answer}. The corresponding area in the image is <mask_start>{self.mask_token}<mask_end>',
                        'mask':mask_path
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'grounded vqa sample nums: {tot}')


    def add_lvis_segmentation_samples(self,lvis_data_root,max_sample_nums=None):
        # import lvis
        # lvis_data = lvis.LVIS(join(lvis_data_root,'lvis_v1_train.json'))
        # ann_ids = lvis_data.get_ann_ids()

        tot = 0
        with open(join(lvis_data_root,'train_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                id = sample['id']
                image_id = sample['image_id']
                category_id = sample['category_id']
                bbox = sample['bbox']
                name = sample['name']
                def_ = sample['def']
                area = sample['area']

                image_path = join(lvis_data_root,'train2017',f'{str(image_id).zfill(12)}.jpg')
                mask_path = join(lvis_data_root,'binary_mask',f'{id}.png')
                # if not os.path.exists(image_path) or not os.path.exists(mask_path):
                #     continue
                if area >= 1000:
                    self.samples.append(
                        {
                            'image_path':image_path,
                            'instruction':f'This is an image:\n<image_start><image><image_end>\nPlease segment out the object that corresponding to the {name}, {def_} in the image.',
                            'output':f'It is <mask_start>{self.mask_token}<mask_end>',
                            'mask_path':mask_path,
                            'task_name':'s4'
                        }
                    )
                    tot += 1
                    if max_sample_nums is not None and tot >= max_sample_nums:
                        break
        
        print(f'lvis segmentation sample nums: {tot}')


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]

        instruction = sample['instruction']
        output = sample['output']
        task_name = sample['task_name']
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            # output = output + '<|im_end|>\n'  # qwen2
            output = output + ' </s>'  # llama2

        data = {
            'instruction':instruction,
            'output':output,
            'task_name':task_name,
        }
        
        image_path = sample.get('image_path',None)
        video = sample.get('video',None)
        audio = sample.get('audio',None)
        mask_path = sample.get('mask_path',None)

        if image_path is not None:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size,self.image_size))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image']=image

        if video is not None:
            vr = VideoReader(uri=video, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['<video>']=video

        if audio is not None:
            audio, sr = librosa.load(audio,sr=16000,mono=True)
            # print('duration: ',len(audio)/sr)
            if len(audio) < sr: # < 1s
                sil = np.zeros(sr-len(audio), dtype=float)
                audio = np.concatenate((audio,sil),axis=0)
            # audio = audio[: 60 * sr]

            window_size = 1 * sr # 1s
            max_duration = len(audio) // window_size
            if len(audio) % window_size != 0:
                max_duration += 1
                pad_length = window_size - len(audio) % window_size
                sil = np.zeros(pad_length,dtype=float)
                audio = np.concatenate((audio,sil),axis=0)
            step = 1
            audio_feature = []
            # print('max_duration: ',max_duration)
            for i in range(0,max_duration,step):
                start = int(i*sr)
                end = int((i + step)*sr)
                audio_seg = torch.from_numpy(audio[start:end]).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['<audio>'] = audio_feature

        if mask_path is not None:
            mask = cv2.imread(mask_path)
            gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            gt_mask = gray_mask > 0
            gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy(gt_mask).to(torch.float32) # (224,224)
            gt_mask = gt_mask.unsqueeze(0) # 1,224,224
            data['mask'] = gt_mask

        return data


class PretrainTestDataset(Dataset):

    def __init__(
        self,
        image_annotation_path='/group/40061/cserdu/data/video-llava/train_json/llava_image_.json',
        video_annotation_path='/group/40061/cserdu/data/video-llava/train_json/valid_valley_.json',
        video_llava_data_root='/group/40061/cserdu/data/video-llava',
        image_caption_task=False,
        video_caption_task=False,
        image_size = 224,
        video_frame_nums = 8,
        # wavcaps_data_root='/DATA/DATANAS2/ligy/WavCaps',
        audiocaps_data_root='/group/40061/cserdu/data/AudioCaps',
        audio_caption_task=False,
        grounded_vqa_data_root='/group/40061/cserdu/data/GroundedVQA',
        segmentation_task=False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        lvis_data_root='/group/40061/cserdu/data/LVIS',
        video_processor: CLIPImageProcessor = None,
        # audio_processor=None,
        tokenizer: transformers.PreTrainedTokenizer = None,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        token_nums = image_scale_nums * token_nums_per_scale
        mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        self.mask_token = ''.join(mask_token)
        print('mask token: ',self.mask_token)
        
        self.samples = []

        if image_caption_task:
            self.add_image_caption_samples(image_annotation_path,video_llava_data_root,max_sample_nums=None)
        
        # if video_caption_task:
        #     self.add_video_caption_samples(video_annotation_path,video_llava_data_root,max_sample_nums=None)
        if video_caption_task:
            pass
            # self.add_video_caption_samples(video_path_list)
    
        if audio_caption_task:
            self.add_audio_caption_samples(audiocaps_data_root,max_sample_nums=None)  

        if segmentation_task:
            # self.add_lvis_segmentation_samples(lvis_data_root,max_sample_nums=None)
            self.add_grounded_vqa_samples(grounded_vqa_data_root,max_sample_nums=None)

        self.video_processor = video_processor
        self.tokenizer = tokenizer


    def add_image_caption_samples(self,image_annotation_path,video_llava_data_root,max_sample_nums=None):
        tot = 0
        with open(image_annotation_path,'r') as f:
            samples = json.load(f)
            for sample in samples:
                image = sample['image']
                image_path = join(video_llava_data_root,image)
                # if not os.path.exists(image_path):
                #     continues
                conversations = sample['conversations']
                instruction = conversations[0]['value']
                instruction = instruction.replace('<image>','<image_start><image><image_end>')
                output = conversations[1]['value']
                if output[-1] not in ['.','!','?']:
                    output += '.'
                self.samples.append(
                    {
                        'image':image_path,
                        'instruction':instruction,
                        'output':output
                    }
                )
                tot+=1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'image caption sample nums: {tot}')

  
    def add_video_caption_samples(self,video_path_list):
        ### custom dataset
        tot = 0
        for vpath in video_path_list:
            self.samples.append(
                {
                    'video':vpath,
                    'instruction':'<video_start><video><video_end>\nPlease describe this video.',
                    'output':'No label.'
                }
            )
            tot += 1
        
        print(f'video caption sample nums: {tot}')


    def add_audio_caption_samples(self,audiocaps_data_root,max_sample_nums=None):
        '''AudioCaps data'''
        tot=0
        with open(join(audiocaps_data_root,'test.json'),'r') as f:
            samples = json.load(f)
            for i,sample in enumerate(samples):
                audiocap_id = sample['audiocap_id']
                start_time = sample['start_time']
                caption = sample['caption']
                audio_path = join(audiocaps_data_root,'data',f'{audiocap_id}.wav')
                self.samples.append(
                    {
                        'audio':audio_path,
                        'instruction':'This is an audio:\n<audio_start><audio><audio_end>\nPlease describe this audio.',
                        'output':caption,
                        'question':'Please describe this audio.',
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'AudioCaps sample nums: {tot}')


    def add_grounded_vqa_samples(self,grounded_vqa_data_root,max_sample_nums=None):
        tot=0
        with open(join(grounded_vqa_data_root,'val_grounding.json'),'r') as f:
            samples = json.load(f)
            for vname,value in samples.items():
                question = value['question']
                most_common_answer = value['most_common_answer']
                image_path = join(grounded_vqa_data_root,'val',vname)
                mask_path = join(grounded_vqa_data_root,'val',vname[:-4]+'.png')
                self.samples.append(
                    {
                        'image_path':image_path,
                        'instruction':f'This is an image:\n<image_start><image><image_end>\nPlease segment out the object that corresponding to the {most_common_answer} in the image.',
                        'output':f'It is <mask_start>{self.mask_token}<mask_end>',
                        'mask_path':mask_path,
                        'task_name':'s4',
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'grounded vqa sample nums: {tot}')


    def add_lvis_segmentation_samples(self,lvis_data_root,max_sample_nums=None):
        tot = 0
        with open(join(lvis_data_root,'train_samples.json'),'r') as f:
            samples = json.load(f)
            for sample in samples:
                id = sample['id']
                image_id = sample['image_id']
                category_id = sample['category_id']
                bbox = sample['bbox']
                name = sample['name']
                def_ = sample['def']
                area = sample['area']

                image_path = join(lvis_data_root,'train2017',f'{str(image_id).zfill(12)}.jpg')
                mask_path = join(lvis_data_root,'binary_mask',f'{id}.png')
                # if not os.path.exists(image_path) or not os.path.exists(mask_path):
                #     continue
                if area >= 1000:
                    self.samples.append(
                        {
                            'image':image_path,
                            'instruction':f'This is an image:\n<image_start><image><image_end>\nWhere is {name}, {def_}',
                            'output':f'It is <mask_start>{self.mask_token}<mask_end>',
                            'mask':mask_path,
                            'question':'Please describe this image.'
                        }
                    )
                    tot += 1
                    if max_sample_nums is not None and tot >= max_sample_nums:
                        break
        
        print(f'lvis segmentation sample nums: {tot}')



    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]

        instruction = sample['instruction']
        output = sample['output']
        task_name = sample['task_name']

        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            # output = output + '<|im_end|>\n'
            output = output + ' </s>'

        data = {
            'instruction':instruction,
            'output':output,
            'task_name':task_name,
        }
        image_path = sample.get('image_path',None)
        video = sample.get('video',None)
        audio = sample.get('audio',None)
        mask_path = sample.get('mask_path',None)

        if image_path is not None:
            data['image_path'] = image_path
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224,224))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['image']=image

        
        if video is not None:
            data['video_path'] = video
            vr = VideoReader(uri=video, height=224, width=224)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = 8
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['<video>']=video

        if audio is not None:
            data['audio_path'] = audio
            audio, sr = librosa.load(audio,sr=16000,mono=True)
            # print('duration: ',len(audio)/sr)
            if len(audio) < sr: # < 1s
                sil = np.zeros(sr-len(audio), dtype=float)
                audio = np.concatenate((audio,sil),axis=0)
            # audio = audio[: 60 * sr]
            window_size = 1 * sr # 1s
            max_duration = len(audio) // window_size
            if len(audio) % window_size != 0:
                max_duration += 1
                pad_length = window_size - len(audio) % window_size
                sil = np.zeros(pad_length,dtype=float)
                audio = np.concatenate((audio,sil),axis=0)
            step = 1
            audio_feature = []
            # print('max_duration: ',max_duration)
            for i in range(0,max_duration,step):
                start = int(i*sr)
                end = int((i + step)*sr)
                audio_seg = torch.from_numpy(audio[start:end]).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['<audio>'] = audio_feature

        if mask_path is not None:
            data['mask_path'] = mask_path
            mask = cv2.imread(mask_path)
            gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            gt_mask = gray_mask > 0
            gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy(gt_mask).to(torch.float32) # (224,224)
            gt_mask = gt_mask.unsqueeze(0)
            data['mask'] = gt_mask
        

        return data



@dataclass
class DataCollatorForPretrainDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names =[]

        for instance in instances:
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            X_modals = {}
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            input_ids = instruction_ids + output_ids
            label = [-100] * len(instruction_ids) + output_ids
        
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image

            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
            
            batch_X_modals.append(X_modals)
        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names,
        }



@dataclass
class DataCollatorForPretrainTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[]
        batch_task_names = []

        for instance in instances:

            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)
            metadata = {
                'instruction':instruction,
                'output':output,
            }
            X_modals = {}
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            if task_name in ['ms3','s4','avss']:
                input_ids = instruction_ids + output_ids
                label = [-100]*len(instruction_ids) + output_ids
            else:
                input_ids = instruction_ids
                label = [-100]*len(instruction_ids)
           
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))

            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                metadata['image_path'] = instance['image_path']
                

            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video
                metadata['video_path'] = instance['video_path']

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance['audio_path']
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
                metadata['mask_path'] = instance['mask_path']
            
            batch_metadata.append(metadata)
            batch_X_modals.append(X_modals)
        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names,
            'batch_metadata':batch_metadata,
        }


def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='train', vpath_list = [],
    image_scale_nums = 2, token_nums_per_scale = 3
):
    if mode == 'train':
        dataset = PretrainDataset(
            image_size=data_args.image_size,
            video_frame_nums=data_args.video_frame_nums,
            image_caption_task=data_args.image_caption_task,
            video_caption_task=data_args.video_caption_task,
            audio_caption_task=data_args.audio_caption_task,
            segmentation_task=data_args.segmentation_task,
            video_processor=image_processor,
            tokenizer=tokenizer,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale,
        )
        data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    
    elif mode == 'test':
        dataset = PretrainTestDataset(
            image_size=data_args.image_size,
            video_frame_nums=data_args.video_frame_nums,
            image_caption_task=data_args.image_caption_task,
            video_caption_task=data_args.video_caption_task,
            audio_caption_task=data_args.audio_caption_task,
            segmentation_task=data_args.segmentation_task,
            video_processor=image_processor,
            tokenizer=tokenizer,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale,
        )

        data_collator = DataCollatorForPretrainTestDataset(tokenizer=tokenizer)
    
    return dataset,data_collator

