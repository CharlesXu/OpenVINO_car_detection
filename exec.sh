#!/bin/bash
python3 lab.py -m model/vehicle-detection-adas-0002.xml \
	-i resources/cars_1900_first_frame.jpg \
	-pt $1
