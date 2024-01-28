import math

from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.models import SRCNN
from scripts.dataloader import *
from .configParser import ConfigParser
from .utils import *


class SrcnnSR:
    def __init__(self, configs="train"):
        title("Initialize")
        self.configs = None
        self.epoch = None
        self.initConfigs(configs)
        self.initParams()

    def initConfigs(self, configs):
        self.configs = configs or self.configs
        self.configs = ConfigParser(self.configs).content
        mkdirs([PATHS.MODELS, PATHS.SCRIPTS, PATHS.SCRIPTS, PATHS.CONFIGS, PATHS.SHELLS, PATHS.CHECKPOINTS,
                PATHS.DATASETS])
        createFiles([PATHS.CONFIG_DEFAULT, PATHS.CONFIG_OVERRIDE])
        if self.configs["usegpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            warn('Using CPU.')

    def trainEpochs(self, start, end):
        self.epoch = start
        self.endEpoch = end
        for epoch in range(start, end):
            self.epoch = epoch
            trainLoss, tranCorrect = self.epochAction("train", self.trainloader)
            self.trainLosses.append(trainLoss.item())
            if (epoch + 1) % self.configs["saveEvery"] == 0:
                self.save()

            validLoss, validCorrect = self.epochAction("valid", self.validloader)
            self.validLosses.append(validLoss.item())
            self.learningRates.append(self.learningRate)
            if validLoss < self.bestValidLoss:
                self.bestValidLoss = validLoss
                [best.unlink() for best in getFiles(self.getCheckpointFolder(), "best*.pth")]  # remove last best pth
                self.save(f"bestEpoch{epoch + 1}.pth")
                info(f"save best model, valid loss {round(validLoss.item(), 3)}")
            self.scheduler.step(validLoss)

    @property
    def learningRate(self):
        return self.optimizer.param_groups[0]['lr']

    def modelForward(self, x, y):
        device = self.device
        #x = x.to(device)
        #y = y.to(device)
        x, y = map(lambda t: rearrange(t.to(device), 'b p c h w -> (b p) c h w'), (x, y))
        out = self.model(x)
        loss = self.criterion(out, y)
        return x, y, out, loss

    def epochAction(self, action, loader):
        isBackward = True if action == "train" else False
        GradSelection = Grad if isBackward else torch.no_grad
        totalLoss, totalCorrect, totalLen = 0, 0, 0
        batchLoader = tqdm(loader)
        if isBackward:
            self.model.train()
        else:
            self.model.eval()
        with GradSelection():
            for x, y in batchLoader:
                self.optimizer.zero_grad()

                device = self.device
                #x = x.to(device)
                #y = y.to(device)
                x, y = map(lambda t: rearrange(t.to(device), 'b p c h w -> (b p) c h w'), (x, y))
                out = self.model(x)
                loss = self.criterion(out, y)

                totalLoss += loss
                totalCorrect += torch.sum(y == out)
                totalLen += len(x)
                if isBackward:
                    loss.backward()
                    self.optimizer.step()
                epochProgress = f"{self.epoch + 1}/{self.configs['epochs']}" if action != "test" else "1/1"
                batchLoader.set_description(
                    desc=f"{action} [{epochProgress}] -lglr {'%.1f' % (-math.log(self.learningRate, 10))} 🕐loss "
                         f"{'%.4f' % (loss.item() / len(y))} ➗loss {'%.4f' % (totalLoss / totalLen)}")
        return totalLoss / len(batchLoader), totalCorrect / len(batchLoader)

    def train(self, loader=None):
        title("Train")
        self.trainloader = loader or self.trainloader
        self.load()
        self.trainEpochs(self.startEpoch, self.configs["epochs"])

    def test(self):
        title("Test")
        self.loadBest()

    def saveObject(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainLosses": self.trainLosses,
            "validLosses": self.validLosses,
            "learningRates": self.learningRates
        }

    def getCheckpointFolder(self):
        return PATHS.CHECKPOINTS / f"X{self.configs['scaleFactor']}" / self.getModelName()

    def getModelName(self):
        return f"SRCNN-lr{self.configs['startLearningRate']}-flip{self.configs['randomFlip']}"

    def save(self, fileName=""):
        epoch = self.epoch
        fileName = fileName or f"epoch{epoch + 1}.pth"
        saveFolder = self.getCheckpointFolder()
        mkdir(saveFolder)
        fileName = saveFolder / fileName
        torch.save(self.saveObject(epoch), fileName)

    def load(self):
        saveFolder = self.getCheckpointFolder()
        startEpoch = self.configs["startEpoch"]

        startEpoch = getFinalEpoch(saveFolder) if startEpoch == -1 else startEpoch  # get real last epoch if -1
        self.startEpoch = startEpoch
        if startEpoch == 0:
            return  # if 0 no load (including can't find )

        modelFile = getFile(saveFolder, f"epoch{startEpoch}.pth")
        self.loadParams(modelFile)

    def loadBest(self):
        modelFile = getFile(self.getCheckpointFolder(), "best*.pth")
        if modelFile:
            self.loadParams(modelFile)
        else:
            warn(f"best model not found under {self.getCheckpointFolder()}\nIs 'bestXXX.pth' exist?")
            self.load()

    def loadParams(self, fileP):
        info(f"load model from {fileP.name}")
        saveObject = torch.load(fileP)
        self.model.load_state_dict(saveObject["model"])
        self.scheduler.load_state_dict(saveObject["scheduler"])
        self.optimizer.load_state_dict(saveObject["optimizer"])
        self.validLosses = saveObject["validLosses"]
        self.trainLosses = saveObject["trainLosses"]
        self.learningRates = saveObject["learningRates"]
        self.bestValidLoss = max([*self.validLosses, 0])

    def initParams(self):
        self.criterion = torch.nn.L1Loss()
        self.model = SRCNN()
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["startLearningRate"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.trainLosses = []
        self.validLosses = []
        self.learningRates = []
        self.bestValidLoss = float("inf")
        self.batchSize = self.configs["batchSize"]
        self.patchSize = self.configs['patchSize']
        self.trainDatasetPath = PATHS.DATASETS / self.configs["datasetPath"]
        self.trainDataset = DIV2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            is_training=True,
            patch_size=self.patchSize,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        self.validDataset = DIV2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            is_training=False,
            patch_size=self.patchSize,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        self.trainloader = DataLoader(
            self.trainDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"])
        self.validloader = DataLoader(
            self.validDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"])


if __name__ == '__main__':
    a = SrcnnSR("train")
