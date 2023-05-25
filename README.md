# Hierarchical Modelling for CO2 Variation Prediction for HVAC System Operation

This code provides the implementation of a hierarchical supervised Machine Learning framework that accurately predicts the CO2 variations over a future time horizon. The code documentation and the different defined steps are in synchrony with the content of the manuscript 
published in **MDPI Algorithms** as part of this journal's **Algorithms for Smart Cities** special issue: 
<br> I. Shaer and A. Shami, “Hierarchical Modelling for CO2 Variation Prediction for HVAC System Operation,” *Algorithms*, vol. 16, no. 5, p. 256, May 2023, doi: 10.3390/a16050256. <br>

Before experimenting with the code, the following steps need to be implemented: 

1. Download the data used for this work, which can be found using this link: https://zenodo.org/record/3774723#.ZGEhAHbMKUl. In this repository, the 
authors provided the `prepare.py` file to produce continuous chunks of data (60 minutes). 
2. Create the `data` folder in the root directory, which is on the same level as `src`. In that directory, create two subdirectories `raw` and `generated`. 
3. Use `prepare.py` to produce the four files corresponding to the training and testing data of room00 and room01 that are used for the experimental procedure described in the paper. 
The resultant files should be stored in the `data/raw` directory. As per the code provided in `prepare.py`, the names of the corresponding files are as follows:
  - `continuous_sections_60_train_room00.pickle`
  - `continuous_sections_60_test_room00.pickle`
  - `continuous_sections_60_train_room01.pickle`
  - `continuous_sections_60_test_room01.pickle`

This paper uniquely uses time-series to image transformation to predict the future direction of CO2 variation predictions. To obtain the data used for our experimental procedure, run the code available in 
`src/make_datasets.py`. The code includes placeholders that allow to generate the image data and the `csv` files used for training our models. 
The image data is obtained using the `generate_images` function while the `csv` files are produced using the `dtGen` class instance. The placeholders that can be altered to obtain the datasets are as follows:
  - `types_data` which is equivalent to `train` or `test`. These values as per the naming convention set in the previous steps.
  - `room_name` which is equivalent to `room01` or `room00`
  - `set_granularities` defines the history and future time windows. 
  - The type of Gramian Angular Field (GAF) mode (`holistic` vs. `local`) can be also changed. 
  
After this operation, the `data/generated` directory should be populated with different folders of the following naming convention: `{room_name}_{type_data}`. In each of these folders,
at least two folders exist, the first includes the GAF images of all the defined features that follows the name of the GAF mode. The second folder titled `model_sets` includes 
`csv` files that will be used for training and testing our methodology. Two examples of the resultant files is as follows:
  - `data/generated/room00_train/local/5_co2/O_co2_0.png` which follows this convention `data/generated/{room_name}_{type_data]/{GAF_model}/{granularity}_{feature_name}`
  - `data/generated/model_sets/h-5_f-5_overlap.csv`

The main code that was used to obtain the results of the paper is found in the `src` root folder. The explanation of each these files is as follows:
1) `train_evaluate_models.py`: is mapped to the results of **Section 6.1** in the paper. The process executed is analogous to hyper-parameter optimization.
2) `best_models_generation.py`: produces the best-perfoming models from the **Section 6.1** using the full trianing set. This procedure produces the results in **Section 6.2**.
3) `Transfer_Learning_direction.py` & `Transfer_Learning_fine_tuning`: constitutes the transfer learning and fine tuning operations explained in **Section 6.3**. 
All of these operations can be executed using the code in `src/driver.py` where we provided a sample code that illustrates how each model is ran. 

# Contact-Info

Please feel free to contact me for any questions or research opportunities. 
- Email: shaeribrahim@gmail.com
- Gihub: https://github.com/ibrahimshaer and https://github.com/Western-OC2-Lab
- LinkedIn: [Ibrahim Shaer](https://www.linkedin.com/in/ibrahim-shaer-714781124/)
- Google Scholar: [Ibrahim Shaer](https://scholar.google.com/citations?user=78fAJ_IAAAAJ&hl=en) and [OC2 Lab](https://scholar.google.com/citations?user=ICvnj9EAAAAJ&hl=en)



