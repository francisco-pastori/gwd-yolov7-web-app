import streamlit as st
from PIL import Image
import subprocess
import sys
import os
import shutil

input_dir = 'temp'
output_dir = 'runs/detect/detections'

def main():
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)
    st.title('Global Wheat Detection with YoloV7')
    st.caption('By Francisco Pastori')
    uploaded_file = st.file_uploader(
        label='Upload an image to perform Wheat Detection:',
        type='jpg'
    )
    if uploaded_file is not None:
        st.image(
            image = uploaded_file.getvalue(),
            caption ='Uploaded Image'
        )
        with open(f'{input_dir}/{uploaded_file.name}', mode='wb') as img:
            img.write(uploaded_file.getvalue())        
        subprocess.run([f'{sys.executable}', 'detect.py', '--weights', 'runs/train/yolov7-custom-wheat/weights/best.pt',
                        '--conf', '0.5', '--img-size', '640', '--source', f'{input_dir}/{uploaded_file.name}',
                        '--no-trace', '--save-txt', '--save-conf', '--name', 'detections'])
        st.image(
            image = Image.open(f'{output_dir}/{uploaded_file.name}'),
            caption ='Processed Image'
        )
        os.remove(f'{input_dir}/{uploaded_file.name}')
        shutil.rmtree(output_dir)

if __name__ == '__main__':
    main()