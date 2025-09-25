from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import VideoMAEForVideoClassification 
from transformers import VideoMAEConfig
import torch
from utils import PreExtractedFeatureDataset, logg
from models import Optimizer
from tqdm import tqdm
import argparse
import yaml
import os 
import logging
from speechbrain.nnet.schedulers import NoamScheduler

# Cấu hình logger


def reload_model(model, optimizer, checkpoint_path):
    """
    Reload model and optimizer state from a checkpoint.
    """
    past_epoch = 0
    path_list = [path for path in os.listdir(checkpoint_path)]
    if len(path_list) > 0:
        for path in path_list:
            if ".ckpt" not in path:
                past_epoch = max(int(path.split("_")[-1]), past_epoch)
        
        load_path = os.path.join(checkpoint_path, f"{model.model_name}_epoch_{past_epoch}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return past_epoch+1, model, optimizer


def train_one_epoch(model, dataloader, optimizer, criterion, device,  scheduler):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        frames = batch['frames'].to(device)  # [B, T, C, H, W]
        labels = batch['label'].to(device)  # [B]

        # print(frames.shape)
        optimizer.zero_grad()

        outputs = model(frames)     # outputs là ImageClassifierOutput
        outputs = outputs.logits     # Tensor [batch_size, num_labels]
        
        loss = criterion(outputs, labels)
        
        loss.backward()

        optimizer.step()

        curr_lr, _ = scheduler(optimizer.optimizer)

        total_loss += loss.item()

        # === In loss từng batch ===
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss, curr_lr


from torchaudio.functional import rnnt_loss

def evaluate(model, dataloader, optimizer, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in enumerate(progress_bar):
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            labels = batch['label'].to(device)  # [B]

            optimizer.zero_grad()

            outputs = model(frames)     # outputs là ImageClassifierOutput
            outputs = outputs.logits     # Tensor [batch_size, num_labels]
            
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(batch_loss=loss.item())

    train_acc = 100. * correct / total


    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average validation loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}%")
    return avg_loss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def main():
    from torch.optim import Adam
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    training_cfg = config['training']
    model_cfg = config['model']
    model_config = VideoMAEConfig(
        num_labels = model_cfg['num_classes'],
        image_size = model_cfg.get('image_size', 224),
        num_frames = model_cfg.get('num_frames', 30),
        patch_size = model_cfg.get('patch_size', 16),
        hidden_size= model_cfg.get('hidden_size', 768),
        num_hidden_layers= model_cfg.get('num_hidden_layers', 1),
        num_attention_heads= model_cfg.get('num_attention_heads', 12),
        intermediate_size= model_cfg.get('intermediate_size', 3072),
        use_mean_pooling= model_cfg.get('use_mean_pooling', True),
    )
    logg(training_cfg['logg'])

    # ==== Load Dataset ====
    train_dataset = PreExtractedFeatureDataset(
        feature_dir = training_cfg['train_path']
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
    )

    dev_dataset = PreExtractedFeatureDataset(
        feature_dir = training_cfg['dev_path']
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoMAEForVideoClassification(
        config=model_config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # add_nan_hook(model)  # Thêm hook để kiểm tra NaN trong model

    criterion = nn.CrossEntropyLoss()


    optimizer = Optimizer(model.parameters(), config['optim'])



    if not config['training']['reload']:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
    else:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
        scheduler.load(config['training']['save_path'] + '/scheduler.ckpt')

    # === Huấn luyện ===

    start_epoch = 1
    if config['training']['reload']:
        checkpoint_path = config['training']['save_path']
        start_epoch, model, optimizer = reload_model(model, optimizer, checkpoint_path)
    num_epochs = config["training"]["epochs"]

    
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, curr_lr = train_one_epoch(model, train_loader, optimizer, criterion, device,  scheduler)
        val_loss = evaluate(model, dev_loader, optimizer, criterion, device)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {curr_lr:.6f}")
        # Save model checkpoint

        model_filename = os.path.join(
            config['training']['save_path'],
            f"{config['model']['model_name']}_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        scheduler.save(config['training']['save_path'] + '/scheduler.ckpt')



if __name__ == "__main__":
    main()

# 3