import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 批量加载整个数据集
folder_path = "/CardRFDataset/CardRF/LOS/Train/UAV/DJI_PHANTOM/FLYING/split/"  # 你的数据集文件夹路径
expand_num = 85400 - 42700  # 需要生成的数据量
current = 11
output_folder = f"/CardRFDataset/CardRF/LOS/Train/UAV/DJI_PHANTOM/FLYING/split/{current}_expanded"  # 生成的数据存放目录

X_list, y_list = [], []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".npz"):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        X_list.append(data["X"])
        y_list.append(data["y"])

# 合并数据
X_all = np.vstack(X_list).astype(np.float32)  # (总样本数, 20480)
y_all = np.concatenate(y_list).astype(np.int32)  # (总样本数,)

print(f"数据加载完成，总样本数: {X_all.shape[0]}, 每个样本维度: {X_all.shape[1]}")

# 3. 归一化数据
X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())  # 归一化到 [0,1]
X_all = torch.tensor(X_all, dtype=torch.float32)

# 4. 定义 VAE
class VAE(nn.Module):
    def __init__(self, input_dim=20480, latent_dim=512):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim * 2)  # 512 维均值 + 512 维方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """ 采样: Z = μ + σ * ε """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_params = self.encoder(x)
        mu, logvar = z_params[:, :512], z_params[:, 512:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# 5. 训练 VAE
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# **优化数据加载**
train_loader = DataLoader(
    TensorDataset(X_all), batch_size=512, shuffle=True
)

# **使用混合精度训练（减少计算量）**
scaler = torch.amp.GradScaler("cuda")

for epoch in range(50):
    for batch in train_loader:
        batch = batch[0].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):  # 启用混合精度
            recon_x, mu, logvar = vae(batch)
            recon_loss = loss_fn(recon_x, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl_loss  # KL 散度权重

        scaler.scale(loss).backward()  # 反向传播
        scaler.step(optimizer)
        scaler.update()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("✅ VAE 训练完成！")

# 6. 训练 Latent Diffusion Model (LDM)
vae.eval()  # 进入推理模式
latent_X = vae.encoder(X_all.to(device))[:, :512].detach()

class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim=512):
        super(LatentDiffusion, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

ldm = LatentDiffusion().to(device)
optimizer_ldm = optim.Adam(ldm.parameters(), lr=1e-3)

for epoch in range(100):
    optimizer_ldm.zero_grad()
    noise = torch.randn_like(latent_X).to(device)

    with torch.amp.autocast("cuda"):
        pred = ldm(noise)
        loss = loss_fn(pred, latent_X)

    scaler.scale(loss).backward()
    scaler.step(optimizer_ldm)
    scaler.update()

    if epoch % 10 == 0:
        print(f"LDM Epoch {epoch}, Loss: {loss.item()}")

print("✅ LDM 训练完成！")

# 7. 生成数据
with torch.no_grad():
    ddim_samples = ldm(torch.randn((expand_num, 512)).to(device))  # 生成新样本
    generated_X = vae.decoder(ddim_samples)  # 还原到 20480 维

# 8. **分块存储数据**
generated_X = generated_X.cpu().numpy()
y_expanded = np.full((expand_num,), 12)

os.makedirs(output_folder, exist_ok=True)

chunk_size = 244
num_chunks = expand_num // chunk_size
if expand_num % chunk_size != 0:
    num_chunks += 1

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, expand_num)

    X_chunk = generated_X[start_idx:end_idx]
    y_chunk = y_expanded[start_idx:end_idx]

    save_path = os.path.join(output_folder, f"{current}_expanded_part{i}.npz")
    np.savez(save_path, X=X_chunk, y=y_chunk)
    print(f"✅ 已保存: {save_path}")

print(f"数据扩展完成，共生成 {num_chunks} 个 .npz 文件，存储在 {output_folder}/ 目录下")

