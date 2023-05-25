from train_evaluate_models import *
from best_models_generation import *
from Transfer_Learning_direction import *
from Transfer_Learning_fine_tuning import *

h_f = "h-5_f-5"
root_dir = ".."
root_folder = "../data/generated"
runs = 3

#Step1: This trains and evaluates models with different hyper-parameters
# cs = Train_Evaluate_Models(
#     h_f,
#     root_dir,
#     "room00",
#     root_folder,
#     runs
# )
# cs.train()

#Step2: This step uses the best performing hyper-parameters obtained from step 1 to produce models on the full training dataset with their performance on the testing set 
# cs = Models_Generation(
#     h_f,
#     root_dir,
#     "room00",
#     root_folder,
#     runs = 1
# )
# cs.train()

#Step3: This step uses the models trained in Step2 to be evaluated in another room (room01), which is a direct application of Transfer Learning
# cs = TransferLearning(
#     h_f,
#     root_dir,
#     "room00",
#     root_folder,
#     runs, 
#     "room01"
# )
# cs.evaluate()

#Step4: This step fine-tunes the models produced in room00 by replacing its ensemble models with the training data gathered in room01
cs = Fine_Tuning(
    h_f,
    root_dir,
    "room00",
    root_folder,
    1,
    "room01"
)
cs.train()