import random
import datetime
import config
import dataset
import engine
import torch
import pandas as pd
import numpy as np
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logger.add("experiment.log")


def run():
    dfx = pd.read_csv('./data/train.csv')
    df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.2, random_state=42)

    def making_label(st):
        if st == 'positive':
            return 1
        elif st == 'neutral':
            return 0
        else:
            return 2

    df_train['sentiment'] = df_train['sentiment'].apply(making_label)
    df_valid['sentiment'] = df_valid['sentiment'].apply(making_label)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # df_test = pd.read_csv('./data/test.csv')

    logger.info(f"Bert Model: {config.BERT_PATH}")
    logger.info(f"Current date and time :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")

    logger.info(f"Train size : {len(df_train):.4f}")
    logger.info(f"Valid size : {len(df_valid):.4f}")

    train_dataset = dataset.BERTDataset(
        review=df_train.content.values,
        target=df_train.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4, shuffle=True)

    valid_dataset = dataset.BERTDataset(
        review=df_valid.content.values,
        target=df_valid.sentiment.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    # model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        logger.info(f"epoch={epoch}")

        train_loss, train_acc = engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler)

        for tag, parm in model.named_parameters():
            if parm.grad is not None:
                writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

        outputs, targets, val_loss, val_acc = engine.eval_fn(
            valid_data_loader, model, device)
        val_mcc = metrics.matthews_corrcoef(outputs, targets)
        logger.info(f"val_MCC_Score = {val_mcc:.3f}")

        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        writer.add_scalar('loss/train', train_loss, epoch)  # data grouping by `slash`
        writer.add_scalar('loss/val', val_loss, epoch)  # data grouping by `slash`

        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        writer.add_scalar('acc/train', train_acc, epoch)  # data grouping by `slash`
        writer.add_scalar('acc/val', val_acc, epoch)  # data grouping by `slash`

        logger.info(f"val_mcc={val_acc:.3f}")
        writer.add_scalar('mcc/val', val_mcc, epoch)  # data grouping by `slash`

        accuracy = metrics.accuracy_score(targets, outputs)
        logger.info(f"Accuracy Score = {accuracy:.3f}")

        if accuracy > best_accuracy:
            print(f"Saving model with Accuracy Score = {accuracy:.3f}")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
