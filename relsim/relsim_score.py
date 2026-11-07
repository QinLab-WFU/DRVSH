import torch
import torch.nn.functional as F
import os
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
from huggingface_hub import hf_hub_download

# ========== 替代 qwen_vl_utils ==========
def process_vision_info(messages):
    images = []
    for msg in messages:
        for content in msg.get('content', []):
            if content.get('type') == 'image':
                images.append(content['image'])
    return images, []   # 视频列表为空
# ======================================

from .relsim_model import QwenWithQueryToken


class relsim:
    def __init__(self, base_model, processor):
        self.base_model = base_model
        self.processor = processor
        self.device = next(base_model.parameters()).device

    def embed(self, img):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "<|query|>"}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # 关键修复：如果视频列表为空，传入 None 而不是空列表
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            embedding = self.base_model(**inputs)
            embedding = embedding.float()
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def __call__(self, img1, img2):
        embedding1 = self.embed(img1)
        embedding2 = self.embed(img2)
        similarity = (embedding1 * embedding2).sum().item()
        return similarity

    def eval(self):
        self.base_model.eval()
        return self


def preprocess_image(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    else:
        raise TypeError("Input must be a PIL Image")


_RelSimClass = relsim


def relsim(pretrained=True, checkpoint_dir=None, cache_dir=None, base_model=None):
    if not pretrained:
        raise ValueError("Only pretrained=True is supported")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided")

    print(f"Loading checkpoint from: {checkpoint_dir}")

    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    base_model_name = base_model if base_model is not None else "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading base model: {base_model_name}")

    # 直接加载到 GPU
    base_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        cache_dir=cache_dir
    )

    print(f"Resizing token embeddings: {len(processor.tokenizer)} tokens")
    base_qwen.resize_token_embeddings(len(processor.tokenizer))

    print("Loading LoRA weights...")
    base_qwen = PeftModel.from_pretrained(
        base_qwen,
        checkpoint_dir
        # 不指定 torch_device，自动继承 GPU
    )

    wrapped_model = QwenWithQueryToken(base_qwen, processor, hidden_size=3584)

    print("Loading projection head...")
    if '/' in checkpoint_dir and not os.path.exists(checkpoint_dir):
        projection_path = hf_hub_download(
            repo_id=checkpoint_dir,
            filename="projection_and_config.pt",
            cache_dir=cache_dir
        )
    else:
        projection_path = os.path.join(checkpoint_dir, "projection_and_config.pt")

    checkpoint = torch.load(projection_path, map_location="cpu")
    wrapped_model.projection.load_state_dict(checkpoint['projection'])

    # 确保投影头也在 GPU
    wrapped_model.projection = wrapped_model.projection.to(wrapped_model.base_model.device)

    print(f"✅ Model loaded on {wrapped_model.base_model.device}")
    wrapped_model.eval()
    print("✅ Model loaded successfully!")

    model = _RelSimClass(wrapped_model, processor)
    return model, preprocess_image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--image1", type=str, required=True)
    parser.add_argument("--image2", type=str, required=True)
    args = parser.parse_args()
    model, preprocess = relsim(pretrained=True, checkpoint_dir=args.checkpoint_dir)
    image1 = preprocess(Image.open(args.image1))
    image2 = preprocess(Image.open(args.image2))
    similarity = model(image1, image2)
    print(f"✅ Similarity score: {similarity:.4f}")