import argparse
import requests
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
ap.add_argument("-n", "--num_images", type=int, default=500, help="# of images to download")
args = vars(ap.parse_args())

# initialize URL that contains the captcha images that will be downloading
# along with the total number of images downloaded thus far
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# loop over the number of images to download
for i in range(0, args["num_images"]):
    try:
        r = requests.get(url, timout=60)    # try to grab a new captcha image - download
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))]) # save to disk
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1

    # handle if any exeptions are thrown during the download process
    except:
        print("[INFO] error downloading image...")

    # insert a small sleep to be courtenous to the server
    time.sleep(0.1)

