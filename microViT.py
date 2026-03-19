import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

# Embedding of the image into patch tokens
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        # conv that produces (B, embed_dim, H/ps, W/ps) then flatten -> sequence
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# Multi-head self-attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP block that follows the multi-head attention in the transformer block
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) # Possibility to add dropout to add redundancy and prevent overfitting

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Transformer block formed by the multi-head attention followed by the MLP
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        # Residual connections to help gradient flow
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# The ViT model: patch embedding, then a stack of transformer blocks, the a classification head
class MiniViT(nn.Module):
    def __init__(self,
                 img_size=32, patch_size=4, in_chans=3,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0,
                 num_classes=10, drop_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Class token
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim)) # Positional embedding
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) # Classification head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        cls = x[:, 0]                         # Extracting the class token
        return self.head(cls)

# Training loop for one epoch
def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        preds = logits.argmax(dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
    return correct / total

def test_random_predictions(model, dataset, device, num_samples=20):
    model.eval()

    indices = random.sample(range(len(dataset)), num_samples)

    print("\nRandom predictions:\n")

    with torch.no_grad():
        for idx in indices:
            img, label = dataset[idx]

            img = img.unsqueeze(0).to(device)
            logits = model(img)
            pred = logits.argmax(dim=1).item()

            true_class = dataset.classes[label]
            pred_class = dataset.classes[pred]

            print(f"expected: {true_class:10s} | predicted: {pred_class}")

# Evaluating model on test dataset
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)

            total += yb.size(0)
            correct += (preds == yb).sum().item()

    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Simple model: 4 heads (and transformer blocks), 0.1 drop rate, 64 embedding dimension and 4 patch size
    model = MiniViT(img_size=32, patch_size=4, embed_dim=64, depth=4, num_heads=4, num_classes=10, drop_rate=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    train_accuracies = []
    test_accuracies = []
    for epoch in range(50):
        train_acc = train_one_epoch(model, train_loader, opt, device)
        test_acc = evaluate(model, test_loader, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"epoch {epoch+1:02d} train acc: {train_acc:.4f} test acc: {test_acc:.4f}")
    
    # Plotting training and test accuracies
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
    
    test_random_predictions(model, test_ds, device, num_samples=20)