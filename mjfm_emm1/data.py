from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .audio_encoder import mock_wav_to_mel

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except Exception:  # pragma: no cover
    HFDataset = None
    load_dataset = None


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetBundle:
    hundred_m: object
    audio_1m: object


class MockTokenizer:
    def __call__(self, text: str, return_tensors: str = 'pt', padding: str = 'max_length', truncation: bool = True, max_length: int = 77):
        tokens = torch.zeros(max_length, dtype=torch.long)
        length = min(len(text.split()), max_length)
        if length > 0:
            tokens[:length] = torch.arange(1, length + 1)
        attention = torch.zeros(max_length, dtype=torch.long)
        attention[:length] = 1
        return {'input_ids': tokens.unsqueeze(0), 'attention_mask': attention.unsqueeze(0)}


def build_image_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


def maybe_load_emm1_datasets(cache_dir: str, allow_mock_data: bool = True) -> DatasetBundle:
    if load_dataset is not None:
        try:
            ds_100m = load_dataset('encord-team/E-MM1-100M', split='train', data_files=['data/nn_01.parquet'], cache_dir=cache_dir)
            ds_1m = load_dataset('encord-team/E-MM1-1M', split='train', data_files='data/audio_to_image.parquet', cache_dir=cache_dir)
            return DatasetBundle(hundred_m=ds_100m, audio_1m=ds_1m)
        except Exception:
            if not allow_mock_data:
                raise
    if not allow_mock_data or HFDataset is None:
        raise RuntimeError('datasets package unavailable and mock data disabled')
    records_100m = {
        'caption': ['a cat on a mat', 'a red car parked', 'mountain landscape at sunset'],
        'file_name_image': ['mock_0.jpg', 'mock_1.jpg', 'mock_2.jpg'],
        'file_name_audio': [None, None, None],
    }
    records_1m = {
        'caption': ['dog barking near a park', 'piano music with audience'],
        'annotation_str': ['good', 'good'],
        'annotated_file_path': ['mock_0.wav', 'mock_1.wav'],
        'paired_file_path': ['mock_0.jpg', 'mock_1.jpg'],
    }
    return DatasetBundle(hundred_m=HFDataset.from_dict(records_100m), audio_1m=HFDataset.from_dict(records_1m))


class EMM1Dataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_dir: str, audio_dir: str, modalities: Sequence[str], image_size: int = 224, max_text_length: int = 77):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.image_dir = Path(image_dir)
        self.audio_dir = Path(audio_dir)
        self.modalities = set(modalities)
        self.max_text_length = max_text_length
        self.image_transform = build_image_transform(image_size)

    def __len__(self):
        return len(self.data)

    def _load_or_mock_image(self, image_name: str | None) -> torch.Tensor:
        if image_name:
            image_path = self.image_dir / image_name
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                return self.image_transform(image)
        return torch.rand(3, 224, 224)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]
        sample: Dict[str, torch.Tensor] = {}
        if 'image' in self.modalities:
            sample['image'] = self._load_or_mock_image(row.get('file_name_image') or row.get('paired_file_path'))
        if 'text' in self.modalities:
            tokens = self.tokenizer(
                row.get('caption', ''),
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
            )
            sample['input_ids'] = tokens['input_ids'].squeeze(0)
            sample['attention_mask'] = tokens['attention_mask'].squeeze(0)
        if 'audio' in self.modalities:
            sample['mel'] = mock_wav_to_mel()
        return sample


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
