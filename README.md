# GestureRNN-ML4Lightpad
Using LSTM's to learn a musician's musical gestures on the Roli Lightpad and collaboratively generate new musical responses

# Checkout the Demo video!
This was originally my final project from Prof. Jason Freeman's "MUSI6002: Interactive Music" class at the Georgia Tech Center for Music Technology.

[![Video link](https://github.com/lamtharnhantrakul/GestureRNN-ML4Lightpad/blob/master/assets/Youtube.png](https://www.youtube.com/watch?v=VgoVGpllaSY)

## *ML4Lightpad* introduction
We use machine learning and deep learning to create a new kind of musical instrument based on the new Roli Lightpad instrument. Firstly, we use Wekinator's machine learning capabilities to continuously interpolate between various sonic parameters in a custom-designed *tension* synthesizer in Ableton Live. More importantly, we train a three-layer LSTM that learns to generate gestures and swipes across the surface of the Lightpad based on user input called *GestureRNN*. GestureRNN regresses continuous values of (x,y) coordinates and instantaneous pressure (p) in real-time based on a user's seed gesture.

<img src="assets/flow_diagram.png" width="900" height="170">

### Rationale

This project explores the notion of using *low dimensional* outputs to create art. Machine learning models used for generative art tend to model and render the final *high dimensional* output directly. For example, [LSTM's](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) trained on MIDI output the final piano roll directly (88x16 ~ 1400 dimensions), [GANs](https://arxiv.org/abs/1708.05509) generate the final image directly (128x128x3 ~ 50000 dimensions) while [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) generates the final waveform directly (22000x256 ~ 5,000,000 dimensions per second).

We argue that humans do not operate in this final high dimensional output. An artist does not think about the final RGB values of each pixel, but instead thinks in terms of brushstrokes and movements across the canvas. A musician does not think of the final score, but instead chooses notes based on [embodied and cognitive process](https://smartech.gatech.edu/handle/1853/58630) that synthesizes the current music, timbre of surrounding instruments and physical constraints of both the player and instrument.

The work was heavily inspired by [David Ha's *SketchRNN*](https://magenta.tensorflow.org/sketch-rnn-demo), a deep learning model trained on human sketches, and conversations about low dimensions of artistic output over Summer 2017 with Doug Eck, both from the Google Magenta team. *SketchRNN* is able to learn pen strokes from over 50,000 sketches and "draw" in real time with the user. *GestureRNN* for Lightpad thus does not generate the final waveform directly, but listens and learns the musical gestures from an experienced player that produce expressive motifs on the instrument.

## Usage Instructions

### System diagram

A high level system diagram of ML4Lightpad system consisting of the Lightpad, MaxMSP, Python and Ableton Live is shown below.

<img src="assets/system_diagram.png" width="800" height="560">

There are many **many** moving parts in this project. I will attempt to explain each component so that if applicable, you can still use parts of this project in your own work.

#### Roli Lightpad
This is a pressure sensitive multi-touch instrument developed by Roli (best known for their futuristic seaboard). I am using the 1st Gen Lightpad; the company just released a [2nd Gen Lightpad M](https://roli.com/products/blocks/lightpad-m?gclid=CjwKCAiAjuPRBRBxEiwAeQ2QPtFx73V8Ysh_XLQu4Td8TKmtWLzx4FsVgizNmRDqLplEiERRFvFOixoCp_0QAvD_BwE). (I'm not sponsored by Roli, I just used it because I own one and it has great LED feedback)

#### MaxMSP
MaxMSP is the brain of the project. The patch is named `ML4Lightpad.patch` in the repo. You will need to download the specialized ["blocks" object](https://cycling74.com/feature/roliblocks) from the Max Package Manager.

#### GestureRNN server
`GestureRNN_server.py` contains the code that receives messages containing (x,y,p) through udp port `8000` from MaxMSP and sends back real-time predictions through udp port `8001` back to Max/MSP. The server also contains a finite state machine to change from `listening` and `predicting` states.

I have attempted to wrap all of this into a docker container defined in `docker.cpu`. To try running this, install docker and run:

`docker run hanoi7/gesture-rnn`

Depending on your system, the udp ports may or may not bind correctly. If anyone knows how to fix this please ping me! I suggest following the method outlined in the next heading where a conda python environment is set up locally on the host machine.

#### Models and python environment

If you want to train your own model, look at the python notebook `GestureRNN_train.ipynb` to see how I loaded data in from MaxMSP and defined/trained the LSTM model. Assuming you are using conda you can run commands:


`conda create --name gesture-rnn python=3.5`
`source activate gesture-rnn`
`pip install -r requirements.txt`
then
`jupyter notebook` to launch a notebook session or
`python GestureRNN_server.py` to launch the server.


Pretrained models with various "lookback" lengths (the number of previous samples used to predict future samples) are available in directory `./models`. The weights are trained on my (Lamtharn Hantrakul) playing style of this instrument. The models do not yet support short taps, it only works on long continuous gestures.

#### Wekinator
[Wekinator](http://www.wekinator.org/) is an awesome machine learning tool by Rebecca Fiebrink. Download it on this link and then use the open file menu to open the `WekinatorProject.wekproj` located in `./GestureRNN_Wekinator`

#### Ableton Live
My DAW of choice for live electronic music performance. The project file is available in the directory `Ableton Project`. I don't use any samples, the synthesis engine is based on the [`tension` physical modelling](https://www.ableton.com/en/packs/tension/) instrument and various effects mapped to macro knobs. If you are an Ableton Live user, this should already be installed as a Live instrument by default.
