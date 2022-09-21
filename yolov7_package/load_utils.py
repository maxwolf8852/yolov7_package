import wget

SCRIPT_16_PATH = 'https://drive.google.com/uc?export=download&id=1L8mPcUvabUscEk6Nr8ck5EFgopgPAMDW&confirm=t'


def load_script_model(out):
	print('Loading torchscript model...')
	wget.download(SCRIPT_16_PATH, out, bar=None)
