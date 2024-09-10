

## **Features**

This python script will load and display all FITS images in a given folder. The main purpopse is to find bad images which shall be removed.


## **Pre-requisites**

Before using this script, ensure you have Python 3.x installed. [Python Installation Instructions](https://python.land/installing-python).   

### **Installation of the blinker.py script**

To install this script, follow these steps:

1. Create a directory to hold the script and associated files

2. Clone or download the repository from my [GitHub repository](https://github.com/photon1503/blinker)

3. To install the required python libraries, navigate to the new directory and run the following command:   

    `pip install -r requirements.txt`



## **Running the Script**

The script is called from the command line. There are three calling methods:


    Linux/MACOS example: python3 blinker.py path-to-fits-folder
    Windows example:     python  blinker.py path-to-fits-folder

Note that you need to quote the path if it contains special characters or spaces.

## Features

1. The script will load all FITS files from the folders and put them in a (maximum) 16GB RAM cache. The cache is big enought to hold thousends of full-frame FITS files.
2. When all images are loaded, they will play at approx 25 frames per second
3. Each image will be stretched using sigma deviations
4. Use the keyboard commands below the naviagate and to remove the bad frames

## Usage

The script is controlled by the keyboard.

 <kbd>A</kbd> Previous image. Press and hold for fast reverse

 <kbd>D</kbd> Next Image. Press and hold for fast forward

  <kbd>R</kbd> Reset, go to first image

 <kbd>Space</kbd> Toggle Play/Pause

 <kbd>P</kbd> Purge image. This will create a "BAD" subfolder and move the images to this location
 
 <kbd>Esc</kbd> Exit script

## **Contributing to blinker.py**

This script is intended for educational purposes in the field of astrophotography. It is part of an open-source project and contributions or suggestions for improvements are welcome.

To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`.
4. Push to the original branch: `git push origin <project_name>/<location>`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).
