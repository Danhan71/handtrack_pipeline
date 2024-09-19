""" To extract all substrokes fora  tiven day, and save
"""

from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper
import sys
from pythonlib.dataset.substrokes import pipeline_wrapper

if __name__=="__main__":
    animal = sys.argv[1]
    DATE = sys.argv[2]

    D = load_dataset_daily_helper(animal, DATE)

    ##### Run main data collection
    Dsubs, DSsubs, SAVEDIR = pipeline_wrapper(D, dosave=True)
