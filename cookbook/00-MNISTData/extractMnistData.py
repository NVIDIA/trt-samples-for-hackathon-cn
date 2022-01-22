import sys
import loadMnistData

nTrain  = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 600
nTest   = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100

mnist = loadMnistData.MnistData("./", isOneHot=False)
mnist.saveImage(nTrain, "./train/", True)   # 60000 images in total
mnist.saveImage(nTest,  "./test/",  False)  # 10000 images in total

