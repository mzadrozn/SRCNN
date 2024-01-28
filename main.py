from fire import Fire
from scripts.srcnnSR import SrcnnSR


def main(config = "train"):
    app = SrcnnSR(config)
    app.train()


if __name__ == '__main__':
    Fire(main)