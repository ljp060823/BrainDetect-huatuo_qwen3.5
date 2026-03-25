import torch
import os
import sys
import tqdm
from torch.utils.data import DataLoader
from model import AttentionUNet
from dataset import MedicalDataset

model = AttentionUNet(n_classes=9).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()  # 多类

best_loss = float('inf')
patience = 10          # 连续 10 次无改进就停止
no_improve = 0 

train_ds = MedicalDataset("/data/unet-attention-dsconv_github/data/train/", "/data/unet-attention-dsconv_github/data/train_mask/")
loader = DataLoader(train_ds, batch_size=14, shuffle=True)

print("开始训练...")

history = []

for epoch in range(15):
    epoch_loss = 0.0
    loader = tqdm.tqdm(loader,desc = f"Epoch {epoch+1}/15")
    for img, mask in loader:
        img, mask = img.cuda(), mask.cuda()
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loader.set_postfix({"loss":f"{loss.item():.4f}"})
    avg_loss = epoch_loss / len(loader)
    history.append(avg_loss)
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        torch.save(model.state_dict(), "unet_atten_dsconv_best.pth")  # 保存最佳
        print(f"Epoch {epoch+1} 损失：{avg_loss:.4f} (新最佳)")
    else:
        no_improve += 1
        print(f"Epoch {epoch+1} 损失：{avg_loss:.4f} (无改进 {no_improve}/{patience})")
    
    if no_improve >= patience:
        print(f"\n 最佳损失：{best_loss:.4f}")
        break
print("训练完成！！！！！！！！！！！！！！！！")