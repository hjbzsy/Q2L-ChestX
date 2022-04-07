import pickle
import random

def read(filepath):
    data = pickle.load(open(filepath, 'rb'))
    print(data.__class__)
    print(len(data))
    # for i in range(2):
    #     print(data[i])
        # print(data[i][0], data[i][-1])


def write(filepath, img_size):
    data = pickle.load(open(filepath, 'rb'))
    random.shuffle(data)
    pickle.dump(data[:1000], open("/home/home/dh/xfan/paper/test/query2labels_original/smalldataset/traindata{}.pickle".format(img_size), "wb"))
    pickle.dump(data[1000:2000], open("/home/home/dh/xfan/paper/test/query2labels_original/smalldataset/testdata{}.pickle".format(img_size), "wb"))


if __name__ == '__main__':
    numLi = [384, 400, 416]
    # numLi = [384]
    for num in numLi:
        print("正在处理：" + str(num))
        filepath = '/home/sda1data/zc/nih/nih/data{}.pickle'.format(num)
        write(filepath, num)
        # read(filepath)
        # print("####################")
    






