import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from huggingface_hub import login
import io

# 🔴 请在这里输入你的Hugging Face token
HF_TOKEN = "hf_xxxx"

def authenticate_huggingface():
    """使用token进行Hugging Face认证"""
    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        login(token=HF_TOKEN)
        print("✅ Hugging Face authentication successful!")
        return True
    else:
        print("⚠️  Please set your HF_TOKEN in the code")
        print("Attempting interactive login...")
        login()
        return True

# ===== 数据集准备 =====
class TextToImageDataset(Dataset):
    def __init__(self, split="train", size=50):
        print(f"Loading dataset split: {split}")
        if not authenticate_huggingface():
            print("Authentication failed, trying public datasets...")
        dataset_options = [
            ("nelorth/oxford-flowers", "train"),
            ("frgfm/imagenette", "train"),
        ]
        self.dataset = None
        for dataset_name, dataset_split in dataset_options:
            print(f"Trying to load {dataset_name} with split '{dataset_split}'...")
            self.dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
            print(f"✅ Successfully loaded {dataset_name}")
            self.dataset_name = dataset_name
            break

        if self.dataset is None:
            print("All datasets failed, creating synthetic dataset...")
            self.dataset = self.create_synthetic_dataset(size)
            self.dataset_name = "synthetic"

        if hasattr(self.dataset, '__len__'):
            available_size = len(self.dataset)
            actual_size = min(size, available_size)
            print(f"Dataset size: {available_size}, using: {actual_size}")
            if hasattr(self.dataset, 'select'):
                self.dataset = self.dataset.select(range(actual_size))
            else:
                self.dataset = self.dataset[:actual_size]
        else:
            print(f"Using full dataset")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.print_dataset_preview()

    def create_synthetic_dataset(self, size):
        print("Creating synthetic dataset...")
        synthetic_data = []
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        shapes = ['circle', 'square', 'triangle', 'rectangle']
        for i in range(size):
            color = colors[i % len(colors)]
            shape = shapes[i % len(shapes)]
            img = Image.new('RGB', (256, 256), color=color)
            caption = f"A {color} {shape} on a white background"
            synthetic_data.append({
                'image': img,
                'caption': caption,
                'text': caption
            })
        return synthetic_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(self.dataset, list):
            item = self.dataset[idx]
        else:
            item = self.dataset[idx]
        image = item['image']
        caption = None
        for key in ['caption', 'text', 'label', 'description']:
            if key in item:
                captions = item[key]
                if isinstance(captions, list):
                    caption = captions[0] if captions else "A generic image"
                else:
                    caption = str(captions) if captions else "A generic image"
                break
        if caption is None:
            if hasattr(self, 'dataset_name'):
                if 'flower' in self.dataset_name.lower():
                    caption = "A beautiful flower"
                elif 'imagenet' in self.dataset_name.lower():
                    caption = "An everyday object"
                else:
                    caption = "An interesting image"
            else:
                caption = "A generic image"
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                image = transforms.ToPILImage()(image)
            else:
                image = Image.new('RGB', (256, 256), color='white')
        image_tensor = self.transform(image)
        return {
            'image': image_tensor,
            'text': caption
        }

    def print_dataset_preview(self, num_samples=5):
        print("\nDataset Preview:")
        for i in range(min(num_samples, len(self.dataset))):
            item = self.dataset[i]
            print(f"Sample {i+1}:")
            if 'image' in item:
                print(f"  Image type: {type(item['image'])}")
                print(f"  Image size: {item['image'].size if hasattr(item['image'], 'size') else 'N/A'}")
            if 'text' in item or 'caption' in item:
                caption = item.get('text') or item.get('caption')
                if isinstance(caption, list):
                    caption = caption[0]
                print(f"  Text: {caption[:70]}...")
            print("-" * 20)
        print("End of preview.")

# ===== 自定义Collate函数 =====
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    return {
        'image': images,
        'text': texts
    }

class UnifiedRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.aesthetic_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.structure_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def get_clip_features(self, images, texts=None):
        with torch.no_grad():
            if images is not None:
                if isinstance(images, list) and isinstance(images[0], Image.Image):
                    inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                else:
                    pil_images = []
                    for img in images:
                        img_normalized = (img + 1.0) / 2.0
                        img_normalized = torch.clamp(img_normalized, 0, 1)
                        if len(img_normalized.shape) == 3 and img_normalized.shape[0] == 3:
                            img_pil = transforms.ToPILImage()(img_normalized.cpu())
                        else:
                            if len(img_normalized.shape) == 4:
                                img_normalized = img_normalized.squeeze(0)
                            img_pil = transforms.ToPILImage()(img_normalized.cpu())
                        pil_images.append(img_pil)
                    inputs = self.clip_processor(images=pil_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(images.device if torch.is_tensor(images) else next(self.clip_model.parameters()).device)
                         for k, v in inputs.items()}
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
            else:
                image_features = None

            if texts is not None:
                text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(next(self.clip_model.parameters()).device) for k, v in text_inputs.items()}
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            else:
                text_features = None
        return image_features, text_features

    def semantic_consistency_reward(self, images, texts):
        image_features, text_features = self.get_clip_features(images, texts)
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        if similarity.dim() == 0:
            similarity = similarity.unsqueeze(0)
        return similarity

    def aesthetic_reward(self, images):
        image_features, _ = self.get_clip_features(images)
        aesthetic_score = self.aesthetic_head(image_features.float())
        aesthetic_score = aesthetic_score.squeeze()
        if aesthetic_score.dim() == 0:
            aesthetic_score = aesthetic_score.unsqueeze(0)
        return aesthetic_score

    def structure_reward(self, images):
        image_features, _ = self.get_clip_features(images)
        structure_score = self.structure_head(image_features.float())
        structure_score = structure_score.squeeze()
        if structure_score.dim() == 0:
            structure_score = structure_score.unsqueeze(0)
        return structure_score

class HierarchicalMixGRPO:
    def __init__(self, model, reward_model, device):
        self.pipe = model
        self.reward_model = reward_model
        self.device = device
        self.early_steps = list(range(0, 15))
        self.mid_steps = list(range(15, 35))
        self.late_steps = list(range(35, 50))
        self.optimizer = optim.AdamW(self.pipe.unet.parameters(), lr=1e-5)
        # 🔴 添加一个用于奖励计算的简化推理步数
        self.reward_inference_steps = 10

    def get_timestep_reward_weights(self, timestep):
        if timestep in self.early_steps:
            return {'semantic': 1.0, 'structure': 0.3, 'aesthetic': 0.1}
        elif timestep in self.mid_steps:
            return {'semantic': 0.5, 'structure': 1.0, 'aesthetic': 0.3}
        else:
            return {'semantic': 0.2, 'structure': 0.5, 'aesthetic': 1.0}

    def compute_hierarchical_reward(self, images, texts, timesteps):
        batch_size = len(images)
        total_rewards = torch.zeros(batch_size).to(images.device)
        
        # 🔴 修复：将图像从 PyTorch tensor 转换为 PIL 图像列表
        pil_images = []
        for img_tensor in images:
            # 去标准化并转换
            img_normalized = (img_tensor + 1.0) / 2.0
            img_normalized = torch.clamp(img_normalized, 0, 1)
            pil_image = transforms.ToPILImage()(img_normalized.cpu())
            pil_images.append(pil_image)

        semantic_rewards = self.reward_model.semantic_consistency_reward(pil_images, texts)
        aesthetic_rewards = self.reward_model.aesthetic_reward(pil_images)
        # 注意：这里结构奖励通常需要更复杂的逻辑，但目前先使用 Aesthetic 头部
        structure_rewards = self.reward_model.structure_reward(pil_images)

        # 确保奖励张量的维度正确
        if semantic_rewards.dim() == 0:
            semantic_rewards = semantic_rewards.unsqueeze(0).expand(batch_size)
        if aesthetic_rewards.dim() == 0:
            aesthetic_rewards = aesthetic_rewards.unsqueeze(0).expand(batch_size)
        if structure_rewards.dim() == 0:
            structure_rewards = structure_rewards.unsqueeze(0).expand(batch_size)
            
        # 修正：由于我们是在一个完整生成图像上计算的奖励，timestep 权重应该针对整个批次是固定的
        # 或者更精确地，我们可以根据训练开始的原始 timestep 来计算权重
        # 为了简单起见，我们假设所有图像都来自同一个阶段
        
        # 🔴 修复：将奖励组件也作为返回值
        return total_rewards, semantic_rewards, aesthetic_rewards

    def train_step(self, batch):
        images = batch['image']
        texts = batch['text']
        self.pipe.unet.train()
        self.optimizer.zero_grad()

        batch_size = images.shape[0]

        # 🔴 改进1: 随机选择一个初始噪声时间步
        initial_timestep = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,), device=self.device)
        
        # 🔴 改进2: 生成图像用于奖励计算
        with torch.no_grad():
            # 为奖励计算运行一个简化的推理循环
            generated_images = self.pipe(
                prompt=texts,
                num_inference_steps=self.reward_inference_steps,
                output_type="pil",
            ).images
            
            # 将PIL图像列表转换为Tensor，因为奖励模型需要它
            reward_images_tensors = [transforms.ToTensor()(img).to(self.device) for img in generated_images]
            reward_images_tensors = torch.stack(reward_images_tensors)
            
            # 计算奖励
            semantic_rewards = self.reward_model.semantic_consistency_reward(generated_images, texts)
            aesthetic_rewards = self.reward_model.aesthetic_reward(generated_images)
            # 结构奖励在这里先用aesthetic替代，或者你有自己的模型
            structure_rewards = self.reward_model.structure_reward(generated_images)
            
            # 🔴 计算总奖励，使用一个固定的阶段权重（例如，中期）
            # 因为整个批次都是在同一个训练步骤生成的，我们选择一个代表性的阶段来应用权重
            weights = self.get_timestep_reward_weights(initial_timestep.item())
            total_rewards = (
                weights['semantic'] * semantic_rewards +
                weights['structure'] * structure_rewards +
                weights['aesthetic'] * aesthetic_rewards
            )

        # 🔴 改进3: 正常执行一步训练
        # 这次使用原始图像和噪声，而不是生成的图像
        latents = self.pipe.vae.encode(images).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        
        # 🔴 Fix: 使用随机时间步来训练UNet
        timesteps_full_batch = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (batch_size,), device=self.device)
        noise_full_batch = torch.randn_like(latents)
        noisy_latents_full_batch = self.pipe.scheduler.add_noise(latents, noise_full_batch, timesteps_full_batch)
        
        text_inputs_full_batch = self.pipe.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        text_embeddings_full_batch = self.pipe.text_encoder(text_inputs_full_batch.input_ids.to(self.device))[0]
        
        noise_pred_full_batch = self.pipe.unet(noisy_latents_full_batch, timesteps_full_batch, text_embeddings_full_batch).sample

        # 计算策略损失
        mse_loss = nn.functional.mse_loss(noise_pred_full_batch, noise_full_batch, reduction='none').mean(dim=[1, 2, 3])
        policy_loss = (mse_loss * (1 - total_rewards)).mean()

        # 反向传播
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pipe.unet.parameters(), 1.0)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'avg_reward': total_rewards.mean().item(),
            'semantic_reward': semantic_rewards.mean().item(),
            'aesthetic_reward': aesthetic_rewards.mean().item(),
        }

    # 其他函数保持不变 ...
    def generate_images_with_intermediate_steps(self, text_prompt, num_inference_steps=50, save_dir="intermediate_generation"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nGenerating image for prompt: '{text_prompt}'")
        generator = torch.Generator(self.device).manual_seed(42)

        latents = torch.randn((1, self.pipe.unet.config.in_channels, 64, 64), device=self.device, generator=generator)
        self.pipe.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            with torch.no_grad():
                text_inputs = self.pipe.tokenizer(text_prompt, padding="max_length", truncation=True, return_tensors="pt")
                text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
                noise_pred = self.pipe.unet(latents, t, text_embeddings).sample

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if (i+1) % 5 == 0 or i == 0 or i == num_inference_steps - 1:
                with torch.no_grad():
                    image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image = Image.fromarray((image * 255).astype(np.uint8))

                    filename = os.path.join(save_dir, f"step_{i+1:03d}.png")
                    image.save(filename)
                    print(f"  - Saved image at step {i+1} to {filename}")

def run_hierarchical_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading pretrained diffusion model...")
    pipe = DiffusionPipeline.from_pretrained("ItsJayQz/GTA5_Artwork_Diffusion", torch_dtype=torch.float32)
    pipe = pipe.to(device)

    reward_model = UnifiedRewardModel().to(device)
    hierarchical_grpo = HierarchicalMixGRPO(
        model=pipe,
        reward_model=reward_model,
        device=device
    )
    print("Preparing dataset...")
    train_dataset = TextToImageDataset(split="train", size=20)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    print("Starting training...")
    num_epochs = 10
    training_logs = []
    for epoch in range(num_epochs):
        epoch_logs = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch['image'] = batch['image'].to(device)
            log = hierarchical_grpo.train_step(batch)
            epoch_logs.append(log)
            if batch_idx % 2 == 0:
                print(f"Batch {batch_idx}: Loss={log['policy_loss']:.4f}, "
                      f"Reward={log['avg_reward']:.4f}, Semantic={log['semantic_reward']:.4f}, Aesthetic={log['aesthetic_reward']:.4f}")
        training_logs.extend(epoch_logs)
        if epoch_logs:
            avg_loss = np.mean([log['policy_loss'] for log in epoch_logs])
            avg_reward = np.mean([log['avg_reward'] for log in epoch_logs])
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Reward: {avg_reward:.4f}")

    print("Generating test samples...")
    test_prompts = [
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A futuristic cityscape at night with flying cars and neon signs, digital art",
        "A detailed painting of a medieval knight in shining armor, a fantasy setting"
    ]
    eval_dir = "evaluation_samples"
    os.makedirs(eval_dir, exist_ok=True)

    for i, prompt in enumerate(test_prompts):
        image = pipe(prompt).images[0]
        image_path = os.path.join(eval_dir, f"generated_sample_{i+1}.png")
        image.save(image_path)
        with open(os.path.join(eval_dir, f"prompt_{i+1}.txt"), "w") as f:
            f.write(prompt)
        print(f"Generated and saved image for prompt: '{prompt}' to {image_path}")

    hierarchical_grpo.generate_images_with_intermediate_steps(test_prompts[0])

    if training_logs:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot([log['policy_loss'] for log in training_logs])
        plt.title('Policy Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.subplot(1, 3, 2)
        plt.plot([log['avg_reward'] for log in training_logs])
        plt.title('Average Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.subplot(1, 3, 3)
        plt.plot([log['semantic_reward'] for log in training_logs], label='Semantic')
        plt.plot([log['aesthetic_reward'] for log in training_logs], label='Aesthetic')
        plt.title('Reward Components')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

    print(f"Experiment completed!")
    print(f"Training logs saved")
    print(f"Evaluation samples saved to {eval_dir}")
    print("Intermediate generation steps saved to the 'intermediate_generation' folder.")

    return hierarchical_grpo, training_logs

# ===== 早期验证实验 =====
def early_validation_experiment():
    print("Running early validation experiment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_model = UnifiedRewardModel().to(device)
    test_dataset = TextToImageDataset(split="train", size=5)

    results = {
        'semantic_scores': [],
        'aesthetic_scores': [],
        'structure_scores': [],
        'texts': [],
        'images': []
    }
    print("Evaluating samples with different reward functions...")
    for i in range(len(test_dataset)):
        item = test_dataset.dataset[i]
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if 'caption' in item:
            captions = item['caption']
        elif 'text' in item:
            captions = item['text']
        else:
            captions = "A sample image"
        if isinstance(captions, list):
            text = captions[0]
        else:
            text = str(captions)
        with torch.no_grad():
            semantic_score = reward_model.semantic_consistency_reward([image], [text])
            aesthetic_score = reward_model.aesthetic_reward([image])
            structure_score = reward_model.structure_reward([image])
        results['semantic_scores'].append(semantic_score.item())
        results['aesthetic_scores'].append(aesthetic_score.item())
        results['structure_scores'].append(structure_score.item())
        results['texts'].append(text)
        results['images'].append(image)
        print(f"Sample {i+1}:")
        print(f"  Text: {text[:50]}...")
        print(f"  Semantic: {semantic_score.item():.3f}")
        print(f"  Aesthetic: {aesthetic_score.item():.3f}")
        print(f"  Structure: {structure_score.item():.3f}")
        print()

    with open('early_validation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'semantic_scores': results['semantic_scores'],
            'aesthetic_scores': results['aesthetic_scores'],
            'structure_scores': results['structure_scores'],
            'texts': results['texts']
        }, f, indent=2, ensure_ascii=False)
    print("Early validation completed. Results saved to early_validation_results.json")
    return results

if __name__ == "__main__":
    print("Hierarchical Reward-Guided MixGRPO Experiment")
    print("=" * 50)
    print("Step 1: Early Validation")
    early_results = early_validation_experiment()
    print("\nStep 2: Full Hierarchical Training")
    model, logs = run_hierarchical_experiment()
    print("\nExperiment pipeline completed!")
    print("Next steps:")
    print("1. Manually evaluate generated images in evaluation_samples/")
    print("2. Compare human rankings with reward model scores")
    print("3. Analyze training curves in training_curves.png")
    print("4. Scale up to larger datasets if validation is successful")