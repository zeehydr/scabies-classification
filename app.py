import argparse
import io
import math
import os
import numpy as np
from PIL import Image
import datetime
import cv2
import torch
from flask import Flask, render_template, request, redirect, url_for, send_file
# from flask_paginate import Pagination

from ultralytics import YOLO

DATETIME_FORMAT =  "%Y%m%d-%H%M%S"

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        print('Post request received')

        all_files = request.files.getlist("file")
        
        for file in all_files:
            if file.filename == '':
                continue

            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            results = model(img)

            for result in results:
                now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
                img_savename = f"static/{now_time}.png"
                print(img_savename)
                result.save(img_savename)
                return render_template("results.html", img_path_original=img_savename , heading="Detected E. Coli Images")

    return render_template("index.html")


@app.route("/detect.html", methods=["GET", "POST"])
def mypredict():
    if request.method == "POST":
        print('Post request received in mypredict()')
        

        all_files = request.files.getlist("file")
        
        for file in all_files:
            if file.filename == '':
                continue

            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            # img = Image.open(img_path)

            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Convert the image to the format expected by the model (ensure it is in RGB)
            if img_array.shape[-1] == 3:  # Check if the image has 3 channels (RGB)
                img_rgb = img_array
            else:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Resize image to be divisible by stride 32 (width and height)
            img_height, img_width = img_rgb.shape[:2]
            new_height = int(np.ceil(img_height / 32) * 32)
            new_width = int(np.ceil(img_width / 32) * 32)
            resized_img = cv2.resize(img_rgb, (new_width, new_height))

            # Convert the image to a tensor and normalize if necessary
            img_tensor = torch.from_numpy(resized_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # Run the model inference on the image tensor
            results = model(img_tensor)

            # results = model(img)

            for result in results:
                now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
                img_savename = f"static/{now_time}.png"
                print(img_savename)
                result.save(img_savename)
                return render_template("detected.html", img_path_original=img_savename , heading="Detected E. Coli Images")
                # return render_template("results.html", img_path_original=img_savename , heading="Detected E. Coli Images")
                

    return render_template("detect.html")


@app.route("/results")
def show_results():
    # Your results rendering code here
    return render_template("results.html")



# @app.route("/detect.html")
# def show_detection():
#     # Your results rendering code here
#     return render_template("detect.html")



@app.route("/contact.html")
def show_contacts():
    # Your results rendering code here
    return render_template("contact.html")


def get_page_args(page_parameter='page', per_page_parameter='per_page'):
    page = request.args.get(page_parameter, 1, type=int)
    per_page = request.args.get(per_page_parameter, 9, type=int)
    return page, per_page, (page - 1) * per_page


@app.route('/images', methods=['GET', 'POST'])
def images():
    # Get the list of files from the static folder.
    static_folder = app.static_folder
    files = os.listdir(os.path.join(app.root_path, static_folder))

    # Calculate the current page, the number of images to display per page,
    # and the starting index of the images to display on the current page.
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    print("page, per_page, offset", page, ", ", per_page, ", ", offset)
    # per_page = 9
    offset = (page - 1) * per_page

    # Determine the current page number.
    page_num = page # Pages are 0-indexed, so add 1.
    prev_num = page -1
    next_num = page + 1
    
    # Get the files to display on the current page.
    files_to_show = files[offset:offset+per_page]

    # Calculate the total number of pages.
    total_pages = int(math.ceil(len(files) / float(per_page)))

    
    # Create a pagination object.
    pagination = Pagination(page=page, per_page=per_page, total=len(files), css_framework='bootstrap4')
    

    # Render the template with the pagination object and the file list.
    return render_template('images.html', files=files_to_show, pagination=pagination, total_pages=total_pages, current_page=page, per_page=per_page, page_num=page_num)




if __name__ == "__main__":
    # model = YOLO("yolov8n.pt")
    model = YOLO("best.pt")
    app.run(host="0.0.0.0")  # debug=True causes Restarting with state





