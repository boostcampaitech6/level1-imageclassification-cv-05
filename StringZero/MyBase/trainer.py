import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, scheduler, device, args, save_dir):
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.model = model.to(device)
        self.save_dir = save_dir
        self.logger = SummaryWriter(log_dir=self.save_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        

    def train(self):
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, ensure_ascii=False, indent=4)


        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(self.args.epochs):
            # Train loop
            self.model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(self.train_loader):# tqdm.. 안써도 ㄱㅊ을듯!
                inputs, _, _, _,labels = train_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outs = self.model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = self.criterion(outs, labels)

                loss.backward()
                self.optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % self.args.log_interval == 0:
                    train_loss = loss_value / self.args.log_interval
                    train_acc = matches / self.args.batch_size / self.args.log_interval
                    current_lr = self.get_lr()
                    print(
                        f"Epoch[{epoch}/{self.args.epochs}]({idx + 1}/{len(self.train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    self.logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(self.train_loader) + idx
                    )
                    self.logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(self.train_loader) + idx
                    )

                    loss_value = 0
                    matches = 0


            # Validation loop
            val_loss, val_acc = self.validate()

            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(self.model.module.state_dict(), f"{self.save_dir}/best.pth")
                best_val_acc = val_acc
                self.update_config_file(best_val_acc)
            torch.save(self.model.module.state_dict(), f"{self.save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            self.logger.add_scalar("Val/loss", val_loss, epoch)
            self.logger.add_scalar("Val/accuracy", val_acc, epoch)
            # self.logger.add_figure("results", figure, epoch)  # Uncomment if figure is needed
            
            self.scheduler.step(val_acc)

    def validate(self):
        self.model.eval()
        val_loss_items = []
        val_acc_items = []

        with torch.no_grad():
            for val_batch in self.val_loader:
                inputs, _, _, _, labels = val_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outs = self.model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = self.criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

        val_loss = np.sum(val_loss_items) / len(self.val_loader)
        val_acc = np.sum(val_acc_items) / len(self.val_loader.dataset)
        return val_loss, val_acc

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        
    def update_config_file(self, best_val_acc):
       config_path = os.path.join(self.save_dir, "config.json")
       # 파일이 존재하면 기존 설정을 읽어오고, 그렇지 않으면 새 딕셔너리를 생성합니다.
       if os.path.exists(config_path):
           with open(config_path, "r", encoding="utf-8") as f:
               config = json.load(f)
       else:
           config = {}
       
       # 최고 검증 점수를 설정에 추가 또는 갱신합니다.
       config['best_val_acc'] = best_val_acc
       
       # 변경된 설정을 파일에 다시 씁니다.
       with open(config_path, "w", encoding="utf-8") as f:
           json.dump(config, f, ensure_ascii=False, indent=4)
           
           
           

class MultiLabelTrainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, scheduler, device, args, save_dir):
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.model = model.to(device)
        self.save_dir = save_dir
        self.logger = SummaryWriter(log_dir=self.save_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.criterion_mask = nn.CrossEntropyLoss()  # mask에 대한 손실 함수
        self.criterion_gender = nn.CrossEntropyLoss()  # gender에 대한 손실 함수
        self.criterion_age = nn.CrossEntropyLoss()
        

    def train(self):
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, ensure_ascii=False, indent=4)

        # criterion_mask = nn.CrossEntropyLoss()  # mask에 대한 손실 함수
        # criterion_gender = nn.CrossEntropyLoss()  # gender에 대한 손실 함수
        # criterion_age = nn.MSELoss()


        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(self.args.epochs):
            # Train loop
            self.model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(self.train_loader):# tqdm.. 안써도 ㄱㅊ을듯!
                inputs, mask_label, gender_label, age_label , labels = train_batch
                inputs = inputs.to(self.device)
                mask_label = mask_label.to(self.device)
                gender_label = gender_label.to(self.device)
                age_label = age_label.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                mask_out, gender_out, age_out = self.model(inputs)
                mask_preds = torch.argmax(mask_out, dim=-1)
                gender_preds = torch.argmax(gender_out, dim=-1)
                age_preds = torch.argmax(age_out, dim=-1)
                preds = 6*mask_preds + 3*gender_preds + age_preds
                # loss = self.criterion(outs, labels)
                # 각 출력에 대한 손실 계산
                loss_mask = self.criterion_mask(mask_out, mask_label)
                loss_gender = self.criterion_gender(gender_out, gender_label)
                loss_age = self.criterion_age(age_out, age_label)
                
                # 손실들을 결합
                loss = loss_mask + loss_gender + loss_age

                loss.backward()
                self.optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % self.args.log_interval == 0:
                    train_loss = loss_value / self.args.log_interval
                    train_acc = matches / self.args.batch_size / self.args.log_interval
                    current_lr = self.get_lr()
                    print(
                        f"Epoch[{epoch}/{self.args.epochs}]({idx + 1}/{len(self.train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    self.logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(self.train_loader) + idx
                    )
                    self.logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(self.train_loader) + idx
                    )

                    loss_value = 0
                    matches = 0


            # Validation loop
            val_loss, val_acc = self.validate()

            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(self.model.module.state_dict(), f"{self.save_dir}/best.pth")
                best_val_acc = val_acc
                self.update_config_file(best_val_acc)
            torch.save(self.model.module.state_dict(), f"{self.save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            self.logger.add_scalar("Val/loss", val_loss, epoch)
            self.logger.add_scalar("Val/accuracy", val_acc, epoch)
            # self.logger.add_figure("results", figure, epoch)  # Uncomment if figure is needed
            
            self.scheduler.step(val_acc)

    def validate(self):
        self.model.eval()
        val_loss_items = []
        val_acc_items = []

        with torch.no_grad():
            for val_batch in self.val_loader:
                inputs, mask_label, gender_label, age_label, labels = val_batch
                inputs = inputs.to(self.device)
                mask_label = mask_label.to(self.device)
                gender_label = gender_label.to(self.device)
                age_label = age_label.to(self.device)
                labels = labels.to(self.device)

                mask_out, gender_out, age_out = self.model(inputs)
                mask_preds = torch.argmax(mask_out, dim=-1)
                gender_preds = torch.argmax(gender_out, dim=-1)
                age_preds = torch.argmax(age_out, dim=-1)
                preds = 6*mask_preds + 3*gender_preds + age_preds

                loss_mask_item = self.criterion_mask(mask_out, mask_label).item()
                loss_gender_item = self.criterion_gender(gender_out, gender_label).item()
                loss_age_item = self.criterion_age(age_out, age_label).item()
                loss_item = loss_mask_item + loss_gender_item + loss_age_item
                
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

        val_loss = np.sum(val_loss_items) / len(self.val_loader)
        val_acc = np.sum(val_acc_items) / len(self.val_loader.dataset)
        return val_loss, val_acc

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        
    def update_config_file(self, best_val_acc):
       config_path = os.path.join(self.save_dir, "config.json")
       # 파일이 존재하면 기존 설정을 읽어오고, 그렇지 않으면 새 딕셔너리를 생성합니다.
       if os.path.exists(config_path):
           with open(config_path, "r", encoding="utf-8") as f:
               config = json.load(f)
       else:
           config = {}
       
       # 최고 검증 점수를 설정에 추가 또는 갱신합니다.
       config['best_val_acc'] = best_val_acc
       
       # 변경된 설정을 파일에 다시 씁니다.
       with open(config_path, "w", encoding="utf-8") as f:
           json.dump(config, f, ensure_ascii=False, indent=4)