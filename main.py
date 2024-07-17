#6 domein addapiton 
# 

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import torch.optim as optim
from src.datasets import ThingsMEGDataset
from src.models import TransformerClassifier
from src.utils import set_seed

def epoch_meg_data(data, events, tmin=-0.5, tmax=1.0, fs=120):
    epochs = []
    for event in events:
        start = int((event + tmin) * fs)
        end = int((event + tmax) * fs)
        if end > data.shape[1]:  # Ensure end is within bounds
            end = data.shape[1]
        epochs.append(data[:, start:end])
    epochs = np.array(epochs)
    
    # ベースライン補正
    baseline = np.mean(epochs[:, :, :int(-tmin * fs)], axis=2, keepdims=True)
    epochs = epochs - baseline
    
    return epochs

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    val_set = ThingsMEGDataset("val", args.data_dir)
    test_set = ThingsMEGDataset("test", args.data_dir)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = TransformerClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学習率スケジューラの定義
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # アーリーストッピングのための変数を定義
    early_stopping_patience = 20  # アーリーストッピングの忍耐エポック数
    early_stopping_counter = 0
    min_val_loss = float('inf')

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            
            class_pred, domain_pred = model(X)
            loss_class = F.cross_entropy(class_pred, y)
            loss_domain = F.cross_entropy(domain_pred, subject_idxs)

            loss = loss_class + args.lambda_domain * loss_domain  # 合計損失
            
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(class_pred, y)
            train_accs.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            with torch.no_grad():
                class_pred, domain_pred = model(X)
            val_loss_class = F.cross_entropy(class_pred, y).item()
            val_loss_domain = F.cross_entropy(domain_pred, subject_idxs).item()
            val_loss = val_loss_class + args.lambda_domain * val_loss_domain

            val_losses.append(val_loss)
            val_accs.append(accuracy(class_pred, y).item())

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {avg_train_loss:.3f} | train acc: {avg_train_acc:.3f} | val loss: {avg_val_loss:.3f} | val acc: {avg_val_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "train_acc": avg_train_acc, "val_loss": avg_val_loss, "val_acc": avg_val_acc})
        
        if avg_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = avg_val_acc

        # アーリーストッピングのチェック
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            cprint("Early stopping triggered.", "red")
            break

        scheduler.step(avg_val_loss)  # エポックの終わりに学習率を更新

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        X = X.to(args.device)
        with torch.no_grad():
            preds.append(model(X)[0].detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()
