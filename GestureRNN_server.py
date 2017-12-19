
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
from collections import deque
from distutils.util import strtobool
import time
import numpy as np
import threading
from distutils.util import strtobool
from keras.models import load_model
import tensorflow as tf

# DEBUG flag
DEBUG = True

# number of samples before LSTM thinks you are interrupting it
interrupt_length = 20

# Load a trained model
graph = tf.get_default_graph() # hack around keras tensorflow in multiple threads
# read https://github.com/fchollet/keras/issues/5896 for more information on the hack
print("loading model...")
with graph.as_default():
    model = load_model('./models/GestureRNN_30.h5')
print("completed model...")

# hack to make state machine in python server work
state = 'listening'
finger_down = 0

def data_handler(unused_addr, args, *osc_args):
    # (args[0], args[1] = (queue, queue_limit)
    # (osc_args[0],osc_args[1],osc_args[2]) = (x_coor, y_coor, pressure)
    queue = args[0]
    queue_limit = args[1]

    data_point = np.array((osc_args[0],osc_args[1],osc_args[2]))
    if len(queue) < queue_limit:
        queue.append(data_point)
    else:
        _ = queue.popleft() # pop oldest value out of queue
        queue.append(data_point) # add newest value to queue
    if DEBUG:
        print("queue len: %s " % len(queue))

    if state != 'playing':
        sendUDPmsg("/queue_length", maxClient, int(len(queue)))

def finger_touch_handler(unused_addr, args, *osc_args):
    # args[0]= queue
    # osc_args[0] = finger_down (true or false) from Max/MSP

    # Change the state of finger_down
    finger_down = int(osc_args[0])
    if DEBUG:
        print("finger_down:",finger_down)
    # empty the queue when user touches lightpad or when they release their finger
    queue = args[0]
    while (len(queue) > 0):
        queue.popleft()

    if state != 'playing':
        sendUDPmsg("/queue_length", maxClient, int(len(queue)))

def player_state_handler(unused_addr, *osc_args):
    # osc_args[0] = player_state (0 or 1) from Max/MSP

    player_state = osc_args[0]
    if player_state == 0:
        global state
        state = 'listening'

        if DEBUG:
            print("switched to listening state")

def sendUDPmsg(address,maxClient,*args):
    msg = osc_message_builder.OscMessageBuilder(address = address)
    for arg in args:
        msg.add_arg(arg)
    msg = msg.build()
    maxClient.send(msg)

def prepSequence(init_sequence):
    pattern = np.array(init_sequence)
    return pattern

def modelPredict(pattern):
    x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
    prediction = model.predict(x, verbose=0) # prediction is (1,3) -> needed for concatenating to next dataset
    return prediction

def appendIdx(idx, datum):
    datum = np.insert(datum, 0, idx) # append index to datapoint
    # `sendUDPmsg()` expects the data to already be strongly typed
    #datum = [int(datum[0]), float(datum[1]), float(datum[2]), float(datum[3])]
    return datum

def sendValues(maxClient,prediction):
    msg = osc_message_builder.OscMessageBuilder(address = '/prediction')
    msg.add_arg(int(prediction[0])) # prepend with an index for Max/MSP `coll` object
    msg.add_arg(float(prediction[1])) # x coor
    msg.add_arg(float(prediction[2])) # y coor
    msg.add_arg(float(prediction[3])) # pressure
    msg = msg.build()
    maxClient.send(msg)

def genSendPredictions(init_sequence):
    pattern = prepSequence(init_sequence)

    start_time = time.time()
    num_predictions = 100
    ramp_length = 20
    with graph.as_default():

        for i in range(num_predictions):
            prediction = modelPredict(pattern) # prediction is (1,3)
            prediction_reshaped = np.squeeze(prediction)
            datum = appendIdx(i, prediction_reshaped)
            #print(datum)
            #sendUDPmsg("/prediction", maxClient, datum)  # reshape to (3,) and send message via UDP immediatly as it is generated
            sendValues(maxClient,datum)

            pattern = np.concatenate((pattern, prediction), axis=0)
            pattern = pattern[1:len(pattern),:]

        (final_x, final_y, final_pressure) = prediction_reshaped
        fade_pressures = np.linspace(final_pressure, 0.0, num=20)

        for j in range(ramp_length):
            ramp_values = np.array((final_x, final_y, fade_pressures[j]))
            ramp_datum = appendIdx(j+num_predictions,ramp_values)
            #sendUDPmsg("/prediction", maxClient, ramp_datum)
            sendValues(maxClient,ramp_datum)
    if DEBUG:
        print("--- LSTM prediction time: %s ---" % (time.time() - start_time))

if __name__ == '__main__':
    # Define the data that will be manipulated by the handlers
    queue = deque()
    queue_limit = 40
    LSTM_lookback = 30

    # Define the server -> Max/MSP port
    maxClient = udp_client.UDPClient('127.0.0.1', 8000)

    # Define the server_thread
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/lightpad_data", data_handler, queue, queue_limit)
    dispatcher.map("/finger_down", finger_touch_handler, queue)
    dispatcher.map("/player_state", player_state_handler)

    server = osc_server.ThreadingOSCUDPServer(('127.0.0.1', 8001), dispatcher)
    serverThread = threading.Thread(target=server.serve_forever)

    # Start the server threads
    serverThread.start()

    # Define data needed for the finite state machine
    init_seq = []

    print(" ------ Start playing! -------")
    while True:
        if state == 'listening':
            if (len(queue) >= LSTM_lookback and finger_down == 0):
                while len(init_seq) < LSTM_lookback: # LSTM_lookback = 30
                    init_seq.append(queue.popleft())
                state = 'predicting'

                if DEBUG:
                    print("switched to predicting state")

        elif state == 'predicting':

            # predict
            genSendPredictions(init_seq)
            init_seq = []  # clear the init_seq
            player_state = 1
            sendUDPmsg("/player_state", maxClient, int(player_state))
            state = 'playing'
            if DEBUG:
                print("switched to playing state")

        elif state == 'playing':
            '''
            the `player_state_handler()` breaks out of this loop when it receives
            a command from Max/MSP that the player has finished playing the sequence
            think of `player_state_handler()` like an arduino interrupt function
            '''

            if len(queue) > interrupt_length:
                # This must be an interruption
                if DEBUG:
                    print("Interrupted!")
                player_state = 0
                sendUDPmsg("/player_state", maxClient, int(player_state))
                state = 'listening'

                if DEBUG:
                    print("switched to listening state")
