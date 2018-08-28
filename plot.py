import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import datetime, time

if len(sys.argv) < 3:
    print("error argument <csv> <graph-description> required")
    sys.exit(-1)

import csv

path = "/users/pgrad/ssinha/PycharmProjects/FirstModel/chart data/model v2 fullimage vs random normal shuffle/"

walltime_wgan=[]
epochs_wgan=[]
value_wgan=[]
#
# walltime_resnet=[]
# epochs_resnet=[]
# value_resnet=[]
#
# walltime_alexnet=[]
# epochs_alexnet=[]
# value_alexnet=[]

walltime_without=[]
epochs_without=[]
value_without=[]

walltime_center=[]
epochs_center=[]
value_center=[]

# walltime_sig=[]
# epochs_sig=[]
# value_sig=[]

# walltime_ssnet=[]
# epochs_ssnet=[]
# value_ssnet=[]


# with open(path+"VGG16untrained/"+sys.argv[1],newline='') as f:
#     r=csv.reader(f)
#     flag=0
#     for line in r:
#         if flag==0:
#             flag=1
#         else:
#             #print(type(line[0]),line[1],line[2])
#             walltime_vgg.append(line[0])
#             epochs_vgg.append(int(line[1]))
#             value_vgg.append(float(line[2]))
#
# with open(path+"Resnetuntrained/"+sys.argv[1], newline='') as f:
#     r = csv.reader(f)
#     flag = 0
#     for line in r:
#         if flag == 0:
#             flag = 1
#         else:
#             #print(type(line[0]), line[1], line[2])
#             walltime_resnet.append(line[0])
#             epochs_resnet.append(int(line[1]))
#             value_resnet.append(float(line[2]))

# with open(path+"Alexnetuntrained/"+sys.argv[1], newline='') as f:
#     r = csv.reader(f)
#     flag = 0
#     for line in r:
#         if flag == 0:
#             flag = 1
#         else:
#             #print(type(line[0]), line[1], line[2])
#             walltime_alexnet.append(line[0])
#             epochs_alexnet.append(int(line[1]))
#             value_alexnet.append(float(line[2]))

# with open(path + sys.argv[3], newline='') as f:
#     r = csv.reader(f)
#     flag = 0
#     for line in r:
#         if flag == 0:
#             flag = 1
#         else:
#             walltime_center.append(line[0])
#             epochs_center.append(int(line[1]))
#             value_center.append(float(line[2]))

with open(path + sys.argv[1], newline='') as f:
    r = csv.reader(f)
    flag = 0
    for line in r:
        if flag == 0:
            flag = 1
        else:
            walltime_wgan.append(line[0])
            epochs_wgan.append(int(line[1]))
            value_wgan.append(float(line[2]))

with open(path + sys.argv[2], newline='') as f:
    r = csv.reader(f)
    flag = 0
    for line in r:
        if flag == 0:
            flag = 1
        else:
            walltime_without.append(line[0])
            epochs_without.append(int(line[1]))
            value_without.append(float(line[2]))

# with open(path+"TenCrop/"+sys.argv[1], newline='') as f:
#     r = csv.reader(f)
#     flag = 0
#     for line in r:
#         if flag == 0:
#             flag = 1
#         else:
#             #print(type(line[0]), line[1], line[2])
#             walltime_sig.append(line[0])
#             epochs_sig.append(int(line[1]))
#             value_sig.append(float(line[2]))


# with open(path + "With/"+sys.argv[1], newline='') as f:
#     r = csv.reader(f)
#     flag = 0
#     for line in r:
#         if flag == 0:
#             flag = 1
#         else:
#             walltime_ssnet.append(line[0])
#             epochs_ssnet.append(int(line[1]))
#             value_ssnet.append(float(line[2]))


    fig = plt.figure()
    # plt.plot(epochs, avgScore, label="Discrete Q-Agent (lr=0.2)", color='b')
    # plt.plot(epochs_leaky, value_leaky, label="weight initialization with normal distribution and L2 Regularization", color='b')
    plt.plot(epochs_without, value_without, label="Modelv2-normal-res32x32-upsampling-2x-randomcrop-noshuffle", color='g')
    # plt.plot(epochs_alexnet, value_alexnet, label="Alexnet "+sys.argv[2], color='r')
    plt.plot(epochs_wgan, value_wgan, label="Modelv2-normal-res32x32-upsampling-2x-randomcrop-shuffle", color='b')
    # plt.plot(epochs_center, value_center, label="Modelv2-normal-res32x32-upsampling-2x-centercrop-noshuffle", color='y')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    plt.savefig("models_dataug_"+sys.argv[1]+".pdf")
    plt.show()


