# 310injectables

## Description
All code used in the Injectables project for ME 310. Two directories are included corresponding to (i) python-based code to run on a computer (for vision software) and (ii) C++ code to run on arduino. These two devices communicate via serial.

## File descriptions

### /310hardware

This directory contains all the Kicad electrical diagrams for MediLoop circuits.

### /vision

The computer vision software runs off a computer that has python installed. It acts as the slave to the arduino master. Each time the arduino sends a 'SNAP' command via serial, the python program takes an image using the connected camera. It processes that image to segment the syringe using an opensource Meta SAM model, and then determines syringe orientation and plunger position. It calculates the number of steps the paddle actuator must move to position the syringe correctly for cutting and sends two bits of information back to the arduino in the data packet form: <d[dir]c[coordinate]> where dir = L or R and c is a number from 0 to the maximum number of steps the paddle can travel in the x-direction.

| Filename            | Filetype       | Description                                        |
|---------------------|----------------|----------------------------------------------------|
| `main.py`           | Python script  | Entry point for the application                    |
| `requirements.txt`  | Text file      | Lists Python dependencies                         |
| `calibrate.py`  | Python script      | Run this file for calibrating the camera. This file saves parameters to 'calibration_data.npz' for an appropriate bounding box of the step before paddle repositioning (where the camera should be taking a photo of the syringe) and indicating where the step gap is located to avoid shadows interfering with plunger detection.                         |
| `calibration_data.npz`  | Numpy data file      | Saves all calibration data for segmentation and plunger detection to function properly.                         |
| `testing_functions/`  | Directory of Python scripts     | These files allow you to test that the camera, segmentation, plunger detection and serial communication are working without running the main script.                         |


### /arduino

The arduino runs on a relatively simple .ino script. It acts as the master to the computer, sending serial commands to the Python program to receive information on how to move the stepper motor controlling the paddle. This paddle repositions the syringe on the final step before cutting.

| Filename            | Filetype       | Description                                        |
|---------------------|----------------|----------------------------------------------------|
| `main/main.ino`           | Arduino script  | Main file to run on Arduino                    |
| `testing_functions/2motors_sync.ino`           | Arduino script  | Run this function to test the two stepper motors which control the aligning and cutting steps                     |
| `testing_functions/2motors_with_limit_switch.ino`           | Arduino script  | Run this function to test the two stepper motors which control the aligning and cutting steps with limit switches                     |


## License
The Injectables 2025
