# Brute CAPTCHA

## Disclaimer
All of this is developed with the intension of using only for educational purpose.

## In order to crack CAPTCHAs, we will have to go through the following steps:

1. Gather CAPTCHAs so we can create labelled data
2. Label the CAPTCHAs to use in a supervised learning model
3. Train our CAPTCHA-cracking CNN
4. Verify and test our CAPTCHA-cracking CNN
5. Export and host the trained model so we can feed it CAPTCHAs to solve
6. Create and execute a brute force script that will receive the CAPTCHA, pass it on to be solved, and then run the brute force attack

### Steps 1–4 start the Docker container
```bash
docker run -d -v /tmp/data:/tempdir/ aocr/full
docker exec -it $(docker ps -q) /bin/bash
cd /ocr/
```

### Create a dataset, Optional steps
```bash
#===Optional steps=====
cd raw_data/dataset/
python3 captcha_examples.py
cd - && cd labelling && python3 labelling.py
cd - 
aocr dataset ./labels/training.txt ./training.tfrecords
#=====================
```

### Training and Testing the CNN
```bash
cd labels && aocr train training.tfrecords
aocr test testing.tfrecords
```

## Hosting Our CNN Model

### Export the weights
```bash
cd /ocr/ && cp -r model /tempdir/

# Complete model's Docker
docker kill $(docker ps -q)

# Run Tensorflow, this will start on on http://localhost:8501/v1/models/ocr/
docker run -t --rm -p 8501:8501 -v /tmp/data/model/exported-model:/models/ -e MODEL_NAME=ocr tensorflow/serving
```