from types import SimpleNamespace


data_configs = SimpleNamespace(**{})

# -----------------------------------OVERALL DATA CONFIG ----------------------
data_configs.datasets_folder = "../../datasets"
data_configs.num_datasets = 5

# -------------------------------------- FOOD 101 CONFIG --------------------------------
food_101 = SimpleNamespace(**{})
food_101.root_folder = "../../datasets/food_101/food-101"
food_101.image_folder = "../../datasets/food_101/food-101/images"
food_101.class_list_path = "../../datasets/food_101/food-101/classes.txt"
food_101.split_info_paths = {
	"train": "../../datasets/food_101/food-101/train.txt",
	"test": "../../datasets/food_101/food-101/test.txt"
}
food_101.preprocessed_data_path = "../../datasets/food_101/preprocessed_data.pt"

# ------------------------------------- OXFORD PETS CONFIG -------------------------------
oxford_pets = SimpleNamespace(**{})
oxford_pets.root_folder = "../../datasets/oxford_pets/oxford-iiit-pet"
oxford_pets.image_folder = "../../datasets/oxford_pets/oxford-iiit-pet/images"
oxford_pets.class_list_path = "../../datasets/oxford_pets/oxford-iiit-pet/annotations/list.txt"
oxford_pets.split_info_paths = {
	"train": "../../datasets/oxford_pets/oxford-iiit-pet/annotations/trainval.txt",
	"test": "../../datasets/oxford_pets/oxford-iiit-pet/annotations/test.txt"
}
oxford_pets.preprocessed_data_path = "../../datasets/oxford_pets/preprocessed_data.pt"

# ----------------------------------- FLOWERS 102 CONFIG ------------------------------------
flowers_102 = SimpleNamespace(**{})
flowers_102.root_folder = "../../datasets/flowers_102/oxford-102-flowers"
flowers_102.image_folder = "../../datasets/flowers_102/oxford-102-flowers/jpg"
flowers_102.split_info_paths = {
	"train": "../../datasets/flowers_102/oxford-102-flowers/train.txt",
	"test": "../../datasets/flowers_102/oxford-102-flowers/test.txt"
}
flowers_102.preprocessed_data_path = "../../datasets/flowers_102/preprocessed_data.pt"

# ---------------------------------- CALTECH 101 CONFIG ------------------------------------
caltech_101 = SimpleNamespace(**{})
caltech_101.root_folder = "../../datasets/caltech_101/101_ObjectCategories"
caltech_101.preprocessed_data_path = "../../datasets/caltech_101/preprocessed_data.pt"

# --------------------------------- STANFORD CARS CONFIG -----------------------------------
stanford_cars = SimpleNamespace(**{})
stanford_cars.root_folder = "../../datasets/stanford_cars/stanford-cars"
stanford_cars.class_list_path = "../../datasets/stanford_cars/stanford-cars/cars_annos.mat"
stanford_cars.train_image_folder = "../../datasets/stanford_cars/stanford-cars/cars_train"
stanford_cars.test_image_folder = "../../datasets/stanford_cars/stanford-cars/cars_test"
stanford_cars.preprocessed_data_path = "../../datasets/stanford_cars/preprocessed_data.pt"

# ------------------------------------- DATASETS CONFIG HOLDER ----------------------------
data_configs.datasets = {
	"food_101": food_101,
	"oxford_pets": oxford_pets,
	"flowers_102": flowers_102,
	"caltech_101": caltech_101,
	"stanford_cars": stanford_cars
}