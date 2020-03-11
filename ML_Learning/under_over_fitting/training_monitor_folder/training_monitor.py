from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0): # one argument followed by 2 optional ones
        # store output path for figure, path to JSON serialized file, and starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath      # path to the output plot that we can use to visualize loss and acc over time
        self.jsonPath = jsonPath    # path to serialize loss and acc values as JSON file - for training history
        self.statAt = startAt       # starting epoch that training is resumed at when using ctrl+c


    def on_train_begin(self, logs={}):
        self.history = {}           # initialize the history dictionary

        # if JSON history path exists, load training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.history = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    for key in self.history.keys():
                        self.history[key] = self.history[key][:self.statAt]

    # epoch - integer representing epoch number, logs - dictionary contains training and validation loss+acc for epoch
    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, acc... for the entire training process
        for (k, v) in logs.items():
            l = self.history.get(k, [])
            l.append(float(v))
            self.history[k] = l

        # check to see if training history should be serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.history))
            f.close()

        # ensure at least two epochs have passed before plotting (epoch starts at zero)
        if len(self.history["loss"]) > 1:
            # plot the training loss and acc
            N = np.arange(0, len(self.history["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.history["loss"], label="train_loss")
            plt.plot(N, self.history["val_loss"], label="val_loss")
            plt.plot(N, self.history["acc"], label="train_acc")
            plt.plot(N, self.history["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.history["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Acc")
            plt.legend()
            # plt.show()
            plt.savefig(self.figPath)
            plt.close()





