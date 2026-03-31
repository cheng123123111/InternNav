#!/usr/bin/env python3

from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
import sys


def install_optional_stubs():
    fake_decord = ModuleType("decord")
    fake_decord.__spec__ = ModuleSpec("decord", loader=None)

    class _VR:
        pass

    fake_decord.VideoReader = _VR
    sys.modules["decord"] = fake_decord

    fake_torchcodec = ModuleType("torchcodec")
    fake_torchcodec.__spec__ = ModuleSpec("torchcodec", loader=None)
    fake_decoders = ModuleType("torchcodec.decoders")
    fake_decoders.__spec__ = ModuleSpec("torchcodec.decoders", loader=None)

    class _VD:
        pass

    fake_decoders.VideoDecoder = _VD
    fake_torchcodec.decoders = fake_decoders
    sys.modules["torchcodec"] = fake_torchcodec
    sys.modules["torchcodec.decoders"] = fake_decoders


def main():
    install_optional_stubs()

    from transformers import AutoProcessor, AutoTokenizer

    from internnav.dataset import internvla_n1_lerobot_dataset as ds

    model_path = "/mnt/data/0923_Interndata/InternNav/checkpoints/InternVLA-N1-DualVLN"
    root = "/mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted"

    ds.data_dict["debug_short10"] = {
        "data_path": root,
        "height": 125,
        "pitch_1": 0,
        "pitch_2": 30,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(model_path)
    args = SimpleNamespace(
        vln_dataset_use="debug_short10",
        model_type="qwen2.5vl",
        image_processor=processor.image_processor,
        sample_step=8,
        predict_step_num=16,
        pixel_goal_only=False,
        num_future_steps=12,
        num_history=8,
        transform_train=None,
        max_pixels=1280,
        min_pixels=256,
        video_max_total_pixels=1664 * 28 * 28,
        video_min_total_pixels=256 * 28 * 28,
    )

    dataset = ds.NavPixelGoalDataset(tokenizer=tokenizer, data_args=args)
    print("dataset_len", len(dataset))

    item0 = dataset[0]
    print("item0_keys", sorted(item0.keys()))
    for key, value in item0.items():
        if hasattr(value, "shape"):
            print("item0", key, tuple(value.shape))
        else:
            print("item0", key, type(value), value)

    collator = ds.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    batch = collator([dataset[0], dataset[1]])
    print("batch_keys", sorted(batch.keys()))
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print("batch", key, tuple(value.shape))
        else:
            print("batch", key, type(value), value)

    print("smoke_test_ok")


if __name__ == "__main__":
    main()
