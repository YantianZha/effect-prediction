# Labing.py
__author__ = "Jianmei Ye"
import os

action = 'putdown'
effect_label_path = "/home/yochan/DL/effect-recognition/dataset/EP2/effect_labels/"
tempInput = effect_label_path + "temp/pour_labels_04.txt"
labelOutput = effect_label_path + "pour/pour_v_04/pour_v_04_"
labelsInput = effect_label_path

labelDict = {'inside':1,'outside':0,\
             'front':2,'back':3,\
             'right':4, 'left':5,'on':6}
def throwLabel():
    f = open(tempInput,'r')
    count = 0
    try:
        while True:
            line = f.readline()
            count += 1
            if not line:
                break
            label = line.split()[0]
            if count < 10:
                labelFile = open(labelOutput+'0'+str(count)+'.txt','w+')
            else:
                labelFile = open(labelOutput+str(count)+'.txt','w+')
            if label == 'in':
                labelFile.write("inside"+"\n")
                labelFile.write("inside(water,bowl)")

            else:
                labelFile.write("outside" + "\n")
                labelFile.write("outside(water,table)")
            labelFile.close()
    finally:
        f.close()


def putdownLabel():
    # print tempInput
    f = open(tempInput, 'r')
    count = 0
    try:
        while True:
            line = f.readline()
            count += 1
            if not line:
                break
            label = line.split()[0]
            print label
            if count < 10:
                labelFile = open(labelOutput + '0' + str(count) + '.txt', 'w+')
            else:
                labelFile = open(labelOutput + str(count) + '.txt', 'w+')
            if label == 'back':
                labelFile.write("back" + "\n")
                labelFile.write("back(cup,bowl)")
            elif label == 'front':
                labelFile.write("front" + "\n")
                labelFile.write("front(cup,bowl)")
            elif label == 'left':
                labelFile.write("left" + "\n")
                labelFile.write("left(cup,bowl)")
            elif label == 'right':
                labelFile.write("right" + "\n")
                labelFile.write("right(cup,bowl)")
            elif label == 'on':
                labelFile.write("on" + "\n")
                labelFile.write("on(cup,bowl)")
            else:
                labelFile.write("inside" + "\n")
                labelFile.write("inside(cup,bowl)")
            labelFile.close()
    finally:
        f.close()

filenames = "/home/yochan/DL/effect-recognition/dataset/EP2/test_filename.txt"

def getGeneralLabel():
    generalFile1 = open("test_ef_labels.txt",'w+')
    nameFile = open(filenames, 'r')
    while True:
        line = nameFile.readline();
        if not line:
            break
        currentf = line.split()[0]
        labelFile = open(effect_label_path + currentf + ".txt")
        head = labelFile.readline();
        clabel = head.split()[0]
        generalFile1.write(`labelDict[clabel]` + '\n')

    generalFile1.close()




if __name__ == '__main__':
    #throwLabel()
    #putdownLabel()
    getGeneralLabel()
