import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model import ExpressionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='checkpoints/expression_model.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"facility: {device}")
    
    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load data - directly using preprocessed data
    train_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'train'), train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'test'), test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    classes = train_dataset.classes
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Save class mapping
    label_map = {i: cls for i, cls in enumerate(classes)}
    
    # Model
    model = ExpressionModel(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        # Add progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Update progress bar with current loss
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        correct, total = 0, 0
        class_correct = {}
        class_total = {}
        
        # Initialize stats for each class
        for i, class_name in enumerate(classes):
            class_correct[i] = 0
            class_total[i] = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Track per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss/len(train_loader):.4f} - Acc: {acc:.4f}")
        
        # Print per-class accuracy
        print("Per-class accuracy:")
        for i, class_name in enumerate(classes):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"  {class_name}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
        print()  # blank line
        
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save({
                'state_dict': model.state_dict(),
                'num_classes': len(classes),
                'label_map': label_map
            }, args.output)
    
    print(f"\nBest accuracy: {best_acc:.4f}")
    print(f"Model saved: {args.output}")


if __name__ == '__main__':
    main()
