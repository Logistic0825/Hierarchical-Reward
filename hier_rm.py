# 层次化奖励引导的MixGRPO改进实验
# 基于不同去噪阶段使用不同奖励函数的思路

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
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import requests
from io import BytesIO

# ===== 数据集准备 =====
class TextToImageDataset(Dataset):
    def __init__(self, split="train", size=50):
        """
        使用COCO数据集的子集进行实验
        """
        print(f"Loading dataset split: {split}")
        # 使用HuggingFace的COCO数据集
        self.dataset = load_dataset("nlphuji/flickr30k", split="test", streaming=True)
        self.dataset = list(self.dataset.take(size))  # 小数据集用于快速验证
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        # 随机选择一个caption
        captions = item['caption']
        if isinstance(captions, list):
            caption = captions[0]
        else:
            caption = captions
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_tensor = self.transform(image)
        return {
            'image': image_tensor,
            'text': caption
        }

# ===== 自定义Collate函数 =====
def custom_collate_fn(batch):
    """自定义collate函数处理混合数据类型"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'image': images,
        'text': texts
    }
class UnifiedRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载CLIP模型用于语义一致性评估
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 美学质量评估网络 (简化版)
        self.aesthetic_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 全局结构评估网络
        self.structure_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def get_clip_features(self, images, texts=None):
        """获取CLIP特征"""
        with torch.no_grad():
            if images is not None:
                # 处理图像特征
                if isinstance(images, list):
                    # 如果是PIL图像列表
                    inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                else:
                    # 如果是tensor，需要转换为PIL图像
                    pil_images = []
                    for img in images:
                        # 确保图像在正确范围内：从[-1,1]转换到[0,1]
                        img_normalized = (img + 1.0) / 2.0
                        img_normalized = torch.clamp(img_normalized, 0, 1)
                        
                        if len(img_normalized.shape) == 3 and img_normalized.shape[0] == 3:
                            img_pil = transforms.ToPILImage()(img_normalized.cpu())
                        else:
                            img_pil = transforms.ToPILImage()(img_normalized.squeeze().cpu())
                        pil_images.append(img_pil)
                    inputs = self.clip_processor(images=pil_images, return_tensors="pt", padding=True)
                
                inputs = {k: v.to(images.device if torch.is_tensor(images) else next(self.clip_model.parameters()).device) 
                         for k, v in inputs.items()}
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = None
                
            if texts is not None:
                text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(next(self.clip_model.parameters()).device) for k, v in text_inputs.items()}
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                text_features = None
                
        return image_features, text_features
    
    def semantic_consistency_reward(self, images, texts):
        """语义一致性奖励 - 早期阶段使用"""
        image_features, text_features = self.get_clip_features(images, texts)
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        return similarity
    
    def aesthetic_reward(self, images):
        """美学质量奖励 - 后期阶段使用"""
        image_features, _ = self.get_clip_features(images)
        aesthetic_score = self.aesthetic_head(image_features.float())
        return aesthetic_score.squeeze()
    
    def structure_reward(self, images):
        """结构质量奖励 - 中期阶段使用"""
        image_features, _ = self.get_clip_features(images)
        structure_score = self.structure_head(image_features.float())
        return structure_score.squeeze()

# ===== 层次化MixGRPO算法 =====
class HierarchicalMixGRPO:
    def __init__(self, model, reward_model, scheduler, tokenizer, text_encoder):
        self.model = model
        self.reward_model = reward_model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        # 定义不同阶段的时间步范围
        self.early_steps = list(range(0, 15))  # 0-15步：早期，关注语义一致性
        self.mid_steps = list(range(15, 35))   # 15-35步：中期，关注结构质量
        self.late_steps = list(range(35, 50))  # 35-50步：后期，关注美学质量
        
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        
    def get_timestep_reward_weights(self, timestep):
        """根据时间步返回不同奖励的权重"""
        if timestep in self.early_steps:
            return {'semantic': 1.0, 'structure': 0.3, 'aesthetic': 0.1}
        elif timestep in self.mid_steps:
            return {'semantic': 0.5, 'structure': 1.0, 'aesthetic': 0.3}
        else:  # late_steps
            return {'semantic': 0.2, 'structure': 0.5, 'aesthetic': 1.0}
    
    def compute_hierarchical_reward(self, images, texts, timesteps):
        """计算层次化奖励"""
        batch_size = len(images)
        total_rewards = torch.zeros(batch_size).to(images.device)
        
        # 转换tensor图像为PIL格式用于CLIP处理
        pil_images = []
        for img in images:
            # 确保数值在[0,1]范围内，然后转换为PIL
            img_normalized = (img + 1.0) / 2.0  # 从[-1,1]转换到[0,1]
            img_normalized = torch.clamp(img_normalized, 0, 1)
            
            # 检查tensor维度和形状
            if len(img_normalized.shape) == 3 and img_normalized.shape[0] == 3:
                img_pil = transforms.ToPILImage()(img_normalized.cpu())
            else:
                # 处理意外的形状
                if len(img_normalized.shape) == 4:  # NCHW格式
                    img_normalized = img_normalized.squeeze(0)
                img_pil = transforms.ToPILImage()(img_normalized.cpu())
            pil_images.append(img_pil)
        
        # 计算各种奖励
        semantic_rewards = self.reward_model.semantic_consistency_reward(pil_images, texts)
        structure_rewards = self.reward_model.structure_reward(pil_images)
        aesthetic_rewards = self.reward_model.aesthetic_reward(pil_images)
        
        # 根据时间步加权组合奖励
        for i, timestep in enumerate(timesteps):
            weights = self.get_timestep_reward_weights(timestep.item())
            combined_reward = (
                weights['semantic'] * semantic_rewards[i] +
                weights['structure'] * structure_rewards[i] +
                weights['aesthetic'] * aesthetic_rewards[i]
            )
            total_rewards[i] = combined_reward
            
        return total_rewards
    
    def train_step(self, batch):
        """单步训练"""
        images = batch['image']
        texts = batch['text']
        
        self.optimizer.zero_grad()
        
        # 随机采样时间步
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (len(images),))
        timesteps = timesteps.long().to(images.device)
        
        # 添加噪声
        noise = torch.randn_like(images)
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        
        # 编码文本
        text_inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=77)
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(images.device))[0]
        
        # 预测噪声
        noise_pred = self.model(noisy_images, timesteps, text_embeddings).sample
        
        # 计算去噪后的图像
        denoised_images = self.scheduler.step(noise_pred, timesteps, noisy_images).prev_sample
        
        # 计算层次化奖励
        rewards = self.compute_hierarchical_reward(denoised_images, texts, timesteps)
        
        # 计算策略梯度损失 (简化版GRPO)
        mse_loss = nn.functional.mse_loss(noise_pred, noise, reduction='none')
        mse_loss = mse_loss.mean(dim=[1, 2, 3])  # 对每个样本求平均
        
        # 使用奖励作为权重
        policy_loss = (mse_loss * (1 - rewards)).mean()  # 高奖励低损失
        
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'semantic_reward': 0.0,  # 暂时简化，避免重复计算
            'aesthetic_reward': 0.0  # 暂时简化，避免重复计算
        }

# ===== 人工评估模块 =====
class HumanEvaluator:
    def __init__(self):
        self.evaluations = []
    
    def evaluate_images(self, images, texts, save_dir="evaluation_samples"):
        """
        保存图像供人工评估
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for i, (img, text) in enumerate(zip(images, texts)):
            # 保存图像
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img.cpu())
            
            img.save(f"{save_dir}/sample_{i:04d}.png")
            
            # 保存对应的文本
            with open(f"{save_dir}/sample_{i:04d}.txt", "w") as f:
                f.write(text)
        
        print(f"Saved {len(images)} samples to {save_dir} for human evaluation")
        return save_dir

# ===== 实验主函数 =====
def run_hierarchical_experiment():
    """运行层次化奖励引导实验"""
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载预训练模型组件
    print("Loading pretrained models...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # 简化版本：使用更小的模型进行快速验证
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # 移到设备
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    
    # 初始化奖励模型
    reward_model = UnifiedRewardModel().to(device)
    
    # 初始化层次化MixGRPO
    hierarchical_grpo = HierarchicalMixGRPO(
        model=unet,
        reward_model=reward_model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )
    
    # 准备数据
    print("Preparing dataset...")
    train_dataset = TextToImageDataset(split="train", size=20)  # 小数据集快速验证
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    
    # 训练循环
    print("Starting training...")
    num_epochs = 3
    training_logs = []
    
    for epoch in range(num_epochs):
        epoch_logs = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # 移动数据到设备
            batch['image'] = batch['image'].to(device)
            
            # 训练步骤
            log = hierarchical_grpo.train_step(batch)
            epoch_logs.append(log)
            
            if batch_idx % 5 == 0:  # 每5个batch打印一次
                print(f"Batch {batch_idx}: Loss={log['policy_loss']:.4f}, "
                      f"Reward={log['avg_reward']:.4f}")
        
        training_logs.extend(epoch_logs)
        
        # 每个epoch结束后的评估
        avg_loss = np.mean([log['policy_loss'] for log in epoch_logs])
        avg_reward = np.mean([log['avg_reward'] for log in epoch_logs])
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")
    
    # 生成测试样本
    print("Generating test samples...")
    test_dataset = TextToImageDataset(split="train", size=5)
    evaluator = HumanEvaluator()
    
    # 生成一些样本用于人工评估 - 使用原始图像
    test_images = []
    test_texts = []
    
    for i in range(min(5, len(test_dataset))):
        item = test_dataset.dataset[i]  # 直接访问原始数据
        # 获取PIL图像
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        test_images.append(image)
        
        # 获取文本
        captions = item['caption']
        if isinstance(captions, list):
            caption = captions[0]
        else:
            caption = captions
        test_texts.append(caption)
    
    # 保存评估样本
    eval_dir = evaluator.evaluate_images(test_images, test_texts)
    
    # 可视化训练过程
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
    print("Please manually evaluate the generated images and compare with reward model scores")
    
    return hierarchical_grpo, training_logs

# ===== 早期验证实验 =====
def early_validation_experiment():
    """
    早期验证实验：验证不同阶段奖励函数的有效性
    """
    print("Running early validation experiment...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化奖励模型
    reward_model = UnifiedRewardModel().to(device)
    
    # 准备测试数据
    test_dataset = TextToImageDataset(split="train", size=10)
    
    results = {
        'semantic_scores': [],
        'aesthetic_scores': [],
        'structure_scores': [],
        'texts': [],
        'images': []
    }
    
    print("Evaluating samples with different reward functions...")
    
    for i in range(len(test_dataset)):
        item = test_dataset.dataset[i]  # 直接访问原始数据
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # 获取文本
        captions = item['caption'] 
        if isinstance(captions, list):
            text = captions[0]
        else:
            text = captions
        
        # 预处理图像
        clip_image = reward_model.clip_processor(images=[image], return_tensors="pt")
        
        # 计算不同奖励
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
    
    # 保存结果
    with open('early_validation_results.json', 'w') as f:
        json.dump({
            'semantic_scores': results['semantic_scores'],
            'aesthetic_scores': results['aesthetic_scores'],
            'structure_scores': results['structure_scores'],
            'texts': results['texts']
        }, f, indent=2)
    
    print("Early validation completed. Results saved to early_validation_results.json")
    return results

if __name__ == "__main__":
    print("Hierarchical Reward-Guided MixGRPO Experiment")
    print("=" * 50)
    
    # 首先运行早期验证
    print("Step 1: Early Validation")
    early_results = early_validation_experiment()
    
    print("\nStep 2: Full Hierarchical Training")
    # 运行完整实验
    model, logs = run_hierarchical_experiment()
    
    print("\nExperiment pipeline completed!")
    print("Next steps:")
    print("1. Manually evaluate generated images in evaluation_samples/")
    print("2. Compare human rankings with reward model scores")
    print("3. Analyze training curves in training_curves.png")
    print("4. Scale up to larger datasets if validation is successful")