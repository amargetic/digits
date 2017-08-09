from dataset import load_convert_mnist

#TODO REPLACE WITH FLAGS
dataset = 'mnist'
dataset_map = {'mnist': load_convert_mnist}

def get_dataset():
	loader = dataset_map[dataset]
	loader.get_data()
