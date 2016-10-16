# Preprocessing

## Usage :

For colour images :
  python src/preprocess_colour.py -i <<input_folder_path>> -o <output_folder_path> -t <<train/test>>

For green channel images :
   python src/preprocess_green.py -i <<input_folder_path>> -o <output_folder_path> -t <<train/test>>

## Options :

  -i : Input folder path, eg: 'input/'
  -o : Output folder path, eg: 'output/'
  -t : String indicating whether images are processed for testing or training, eg: 'train'