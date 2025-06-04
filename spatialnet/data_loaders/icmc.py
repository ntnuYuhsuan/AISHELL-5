# the HW recorded in car dataset used

import json
import os
from os.path import *
import random
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import librosa
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset
from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.diffuse_noise import (gen_desired_spatial_coherence, gen_diffuse_noise)
from data_loaders.utils.window import reverberation_time_shortening_window
from data_loaders.utils.audio_process import trunc_to_len_multichannel
from data_loaders.utils.drc import drc_multichannel

def enrollments_pool(data_list):
    enroll_pool = {}
    for pair in data_list:
        try:
            pth, spk_id, zone_id = pair.split(' ')
            if zone_id not in enroll_pool.keys():
                enroll_pool[zone_id] = []
                enroll_pool[zone_id].append(pair)
            else:
                enroll_pool[zone_id].append(pair)
        except Exception as e:
            print('error pair:', pair)
    return enroll_pool


class InCarDataset(Dataset):

    def __init__(self,
                 data_conf,
                 repeat=1,
                 seg_length=1,
                 num_samples=10000,
            ):
        with open(data_conf, 'r') as fid:
            config = json.load(fid)

        # Load the original speech list
        with open(config['speech_list'], 'r') as fid:
            self.speech_list = [line.strip() for line in fid.readlines()]
        self.speech_list *= repeat
        
        self.id_list = {}
        # Load the near speech list (for training labels)
        with open(config['speech_list_near'], 'r') as lf:
            for line in lf.readlines():
                id, label_wav_path = line.strip().split(' ')
                self.id_list[id] = label_wav_path

        with open(config["noise_list"], 'r') as fid:
            self.noise_lst = [line.strip() for line in fid.readlines()]
        
        self.id_list = {}
        with open(config['speech_list'], 'r') as fid:
            for line in fid.readlines():
                pth, spk_id, zone_id = line.strip().split(' ')
                wav_name = pth.split('/')[-1]
                id = wav_name.split('.')[0]
                self.id_list[id] = pth

        # 建立座位与音频的映射表
        self.sit_dict = enrollments_pool(self.speech_list)
        self.seg_length = int(seg_length * 16000)
        self.seg_length = 16000
        
        self.randstates = [np.random.RandomState(idx) for idx in range(3000)]
        self.num_samples = num_samples
        # self.use_noise = False

    def __len__(self):
        # 长度为每轮训的步数
        return self.num_samples
    
    def load_wav(self, path, target_sr=16000):
        sig, sr = sf.read(path)
        # 如果采样率不是16k，则重采样到16k
        if sr != target_sr:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
        return sig.T
    
    def choice(self, data_list: list):
        data_list_len = len(data_list)
        choice_idx = random.randint(0, data_list_len - 1)
        pth = data_list[choice_idx]
        return pth
    
    def random_scaling(self, audio, db_range=5):
        scale_factor = 10 ** (np.random.uniform(-db_range, db_range) / 20)
        audio = audio * scale_factor
        return audio
    
    def up_scale(self, audio, scale):
        audio = audio * scale
        return audio

    def highpass(self, audio):
        highpass = 200 // 2
        nyq = 0.5 * 16000
        wn = highpass / nyq
        order = 5
        sos = butter(order, wn, btype='highpass', output='sos')

        if audio.ndim == 2:
            num_channels = audio.shape[0]
            filtered_data = np.empty(audio.shape, dtype=audio.dtype)
            for i in range(num_channels):
                filtered_data[i, :] = sosfiltfilt(sos, audio[i, :])
        else:
            filtered_data = sosfiltfilt(sos, audio)
        return filtered_data
    
    def get_num_spk(self):
        tmp = random.random()
        if tmp < 0.7:
            num_spk = 1
        else:
            num_spk = 2
        return num_spk

    def generate_spk_sit(self, num_spk):
        spk_list = np.arange(4)
        random.shuffle(spk_list)
        return np.sort(spk_list[:num_spk])



    def __getitem__(self, index_seed: tuple[int, int]):
        index, seed = index_seed
        num_spk = self.get_num_spk()
        spk_sit_list = self.generate_spk_sit(num_spk)
        spk_sit_map = np.zeros(4)
        spk_sit_map[spk_sit_list] = 1
        speechs = []
        exits_spks = set()
        label_4channel = np.zeros((4, self.seg_length))



        for sit in spk_sit_list:
            speech_item = self.choice(self.sit_dict[str(sit + 1)]).split(" ")
            speech_path, spk_id, zone_id = speech_item
            while spk_id in exits_spks:
                speech_item = self.choice(self.sit_dict[str(sit + 1)]).split(" ")
                speech_path, spk_id, zone_id = speech_item

            exits_spks.add(spk_id)
            speech = self.load_wav(speech_path)
            speech = trunc_to_len_multichannel(speech, self.seg_length)
            speech = speech.reshape(4, 1, self.seg_length)
            speech = speech[:, 0, :]
            speech = self.up_scale(speech, scale=16)
            speech = self.random_scaling(speech, db_range=5)
            speech = self.highpass(speech)

            new_order = [0, 1, 2, 3]
            speech = speech[new_order, :]
            speechs.append(speech)

            # 利用近场语音数据作为label
            speech_id = speech_path.split('/')[-1].split('.')[0]
            label_wav_path = self.id_list[speech_id]
            label_speech = self.load_wav(label_wav_path)
            
            label_speech = np.tile(label_speech, (4, 1))
            label_speech = trunc_to_len_multichannel(label_speech, self.seg_length)
            speech = speech.reshape(4, 1, self.seg_length)
            speech = speech[:, 0, :]
            new_order = [0, 1, 2, 3]
            speech = speech[new_order, :]
            label_4channel[sit] = speech[sit]

        # Mixing the noise
        # if self.use_noise:
        #     noise_path = self.choice(self.noise_lst)
        #     noise = self.load_wav(noise_path)
        #     noise = trunc_to_len_multichannel(noise, self.seg_length)
        #     noise = noise.reshape(4, 1, self.seg_length)
        #     noise = noise[:, 0, :]
        #     noise = self.up_scale(noise, scale=16)
        #     noise = self.random_scaling(noise, db_range=10)
        #     new_order = [0, 1, 2, 3]
        #     noise = noise[new_order, :]

        #     mix = np.zeros((4, self.seg_length))
        #     for speech in speechs:
        #         mix += speech
        #     mix += noise
        # else:
        #     mix = speechs
        mix = np.zeros((4, self.seg_length))
        mix = speechs
        mix = np.array(mix)
        paras = {
            'sample_rate': 16000,
        }

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(label_4channel, dtype=torch.float32), paras



class InCarTestDataset(Dataset):
    '''
    这个类与 InCarDataset 类似，但它用于处理测试数据。
    与训练数据集的区别在于，它只加载单个音频文件，不进行音频混合操作，只返回一个音频和相关参数。
    '''
    def __init__(self, data_conf):
        with open(data_conf, 'r') as fid:
            config = json.load(fid)
        self.test_data = config['test_data']
        with open(self.test_data, 'r') as fid:
            self.test_data = [line.strip() for line in fid.readlines()]

    def __len__(self):
        return len(self.test_data)
    
    def load_wav(self, path, target_sr=16000):
        sig, sr = sf.read(path)
        # 如果采样率不是16k，则重采样到16k
        if sr is not target_sr:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
        return sig.T
    
    def up_scale(self, audio, scale):
        audio = audio * scale
        return audio
    
    def drc(self, audio, sr):
        result = drc_multichannel(audio, sr)
        return result
    
    def highpass(self, audio):
        highpass = 200 // 2
        nyq = 0.5 * 16000
        wn = highpass / nyq
        order = 5
        sos = butter(order, wn, btype='highpass', output='sos')

        # audio shape: [num_channels, time_length]
        if audio.ndim == 2:
            num_channels = audio.shape[0]
            filtered_data = np.empty(audio.shape, dtype=audio.dtype)
            for i in range(num_channels):
                filtered_data[i, :] = sosfiltfilt(sos, audio[i, :])
        else:
            filtered_data = sosfiltfilt(sos, audio)
        return filtered_data
    
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index_seed: tuple[int, int]):
        index, seed = index_seed
        path = self.test_data[index]
        utt_id = path.split("/")[-1]
        data = self.load_wav(path).astype(np.float32)
        
        T = data.shape[-1]
        mix_noisy = data.reshape(4, 1, T)
        new_order = [3, 1, 2, 0]
        mix_noisy = mix_noisy[new_order, :, :]

        data = []
        for area in range(4):
            data.append(mix_noisy[area][0])

        data = np.stack(data, 0)
        # data = np.stack(mix_noisy, 0)

        # max_norm = np.max(np.abs(data))
        # if max_norm == 0:
        #     max_norm = 1
        # data = data / max_norm
        # data = self.up_scale(data, 16)
        data = self.drc(data, 16000)
        data = self.highpass(data)
        data = data.astype(np.float32)
        # inputs = torch.from_numpy(data)

        paras = {
            'index': index,
            'utt_id': utt_id,
            'sample_rate': 16000,
        }
        
        return torch.as_tensor(data, dtype=torch.float32), paras

class InCarDataModule(LightningDataModule):

    def __init__(
        self,
        datasets: List[str] = ["train", "val", "test"],
        data_conf: Tuple[str, str, str] = ["~/data/config_tr.json", "~/data/config_tr.json", "~/data/config_test.json"],
        batch_size: List[int] = [1, 1, 1],  # batch size for [train, val, {test, predict}]
        seg_length: Tuple[Optional[int], Optional[int], Optional[int]] = [4, 4, None],
        num_samples: List[int] = [10000, 1500],
        num_workers: int = 16,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int] = [None, 2, 2],  # random seeds for train/val
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_conf = data_conf
        self.seg_length = seg_length
        self.num_samples = num_samples
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 3:
            self.batch_size.append(1)

        rank_zero_info("dataset: hw_recorded")
        rank_zero_info(f'batch size: train/val = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val = {self.seg_length}')

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, data_conf, seg_length, num_samples, seed, shuffle, batch_size, collate_fn):
        
        ds = InCarDataset(
            data_conf=data_conf,
            seg_length=seg_length,
            num_samples=num_samples,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
    
    def construct_test_dataloader(self, data_conf, seed, shuffle, batch_size, collate_fn) -> DataLoader:
        ds = InCarTestDataset(
            data_conf=data_conf,
        )
        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
    
    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            data_conf=self.data_conf[0],
            seg_length=self.seg_length[0],
            num_samples=self.num_samples[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            data_conf=self.data_conf[1],
            seg_length=self.seg_length[1],
            num_samples=self.num_samples[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_test_dataloader(
            data_conf=self.data_conf[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            ovlp=self.ovlp[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    """python -m data_loaders.sms_wsj_plus"""
    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(InCarDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='predict')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 1  # for debuging
    datamodule = InCarDataModule(**args_dict)
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    if type(dataloader) != dict:
        dataloaders = {args.dataset: dataloader}
    else:
        dataloaders = dataloader

    for ds, dataloader in dataloaders.items():

        for idx, (noisy, tar, paras) in enumerate(dataloader):
            print(f'{idx}/{len(dataloader)}', end=' ')
            if idx > 10:
                continue
            # write target to dir
            if args.gen_target and not args.dataset.startswith('predict'):
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(tar[0, :, 0, :].numpy())) <= 1
                for spk in range(tar.shape[1]):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, tar[0, spk, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write unprocessed's 0-th channel
            if args.gen_unprocessed:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(noisy[0, 0, :].numpy())) <= 1
                for spk in range(len(paras[0]['saveto'])):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, noisy[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            print(noisy.shape, None if args.dataset.startswith('predict') else tar.shape, paras)
            
            