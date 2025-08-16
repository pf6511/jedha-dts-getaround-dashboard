
import pandas as pd
import requests

import os
from pathlib import Path
import sys

import boto3

def load_data_to_s3():
    try:
        session = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), 
                                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
        s3 = session.resource("s3")
        bucket = s3.Bucket("pfe-getaround-bucket") 
        base_dir = os.path.dirname(__file__)
        DATA_FOLDER_PATH = "data"
        FILE_NAME = 'get_around_delay_analysis.xlsx'
        DELAY_ANALYSIS_XL_FILE_PATH = os.path.join(base_dir, DATA_FOLDER_PATH,FILE_NAME)

        bucket.upload_file(DELAY_ANALYSIS_XL_FILE_PATH, "".join([DATA_FOLDER_PATH, "/",FILE_NAME]))
        print(f" ✅ {DELAY_ANALYSIS_XL_FILE_PATH} uploaded successfully")
    except Exception as e:
        print(f"❌ Failed to upload {DELAY_ANALYSIS_XL_FILE_PATH} : {e}")    

if __name__ == "__main__":
    load_data_to_s3()