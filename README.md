Please download the dataset to form the structure of 'data' folder:
\data
|--\dataset_name
|----{original data files contained in TUDataset}

Once the dataset is downloaded and properly structured, you can run 'data_preprocessing.sh' in 'scripts' folder. To process the specific dataset you downloaded, please see the commands listed in 'data_preprocessing.sh'.

After the data preprocessing, please specify the dataset in 'main.py' in 'codes' folder and run 'negative_sample_generation.sh' in 'scripts' folder.

Finally, you can run the programme by running 'scripts.sh' in 'codes' folder. The settings can be customized in 'main.py'. All the setting options are listed following the main function of 'main.py'.