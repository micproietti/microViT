import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Import the core model and training functions from your base file
from microViT import MiniViT, train_one_epoch, evaluate, test_random_predictions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Trying multiple models variations (drop rates, patch sizes, depth) and comparing them train and test accuracies over epochs to find the best one
    drop_rates = [0.0, 0.1, 0.2]
    patch_sizes = [4, 8]
    depths = [4, 6]
    models = {}
    
    for drop_rate in drop_rates:
        for patch_size in patch_sizes:
            for depth in depths:
                model_name = f"drop{drop_rate}_patch{patch_size}_depth{depth}"
                models[model_name] = MiniViT(
                    img_size=32, 
                    patch_size=patch_size, 
                    embed_dim=64, 
                    depth=depth, 
                    num_heads=4, 
                    num_classes=10, 
                    drop_rate=drop_rate
                ).to(device)

    train_accuracies = {}
    test_accuracies = {}
    optimizers = {}
    
    for model_name, model in models.items():
        optimizers[model_name] = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        train_accuracies[model_name] = []
        test_accuracies[model_name] = []

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        opt = optimizers[model_name]
        for epoch in range(50):
            train_acc = train_one_epoch(model, train_loader, opt, device)
            test_acc = evaluate(model, test_loader, device)

            train_accuracies[model_name].append(train_acc)
            test_accuracies[model_name].append(test_acc)

            print(f"{model_name} | epoch {epoch+1:02d} train: {train_acc:.4f} test: {test_acc:.4f}")
    
    # Now plotting the training and test accuracies for all model variations
    plt.figure(figsize=(14,7))

    model_names = list(models.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, model_name in enumerate(model_names):
        color = colors[i]

        # Training accuracy (solid line)
        plt.plot(
            train_accuracies[model_name],
            color=color,
            linewidth=2,
            linestyle='-',
            label=f"{model_name} train"
        )

        # Test accuracy (dashed line)
        plt.plot(
            test_accuracies[model_name],
            color=color,
            linewidth=2,
            linestyle='--',
            label=f"{model_name} test"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy for MiniViT Variants")

    plt.grid(alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()

    # Picking the best model based on test accuracy and showing some random predictions
    best_model_name = max(test_accuracies, key=lambda name: max(test_accuracies[name]))
    print(f"\nBest model: {best_model_name} with test accuracy: {max(test_accuracies[best_model_name]):.4f}")
    
    best_model = models[best_model_name]
    test_random_predictions(best_model, test_ds, device, num_samples=20)