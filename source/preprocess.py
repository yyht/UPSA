import csv , sys
import utils
import random

def reproducefile(filename):
    trainpath = 'data/twitter.txt'
    train_object = open(trainpath, 'w')
    text2id = {}
    with open(filename) as f:
        for line in f:
            text = line.split('\t')
            # print(text[0])
            assert '.' in text[0] or text[0].strip()=='NaN'
            assert len(text)==3
            # s = text[1]
            # s1=text[2]
            s = utils.clarify(text[1]).lower()
            s1 = utils.clarify(text[2]).lower()
            slen = len(s.split())
            s1len = len(s1.split())
            if not text2id.has_key(s) and slen<15 and slen>1:
                text2id[s] = 1
            if not text2id.has_key(s1) and s1len<15  and s1len>1:
                text2id[s1] = 1
    for x in text2id.keys():
        train_object.write(x)
    train_object.close( )

def reproducetest(filename):
    trainpath = 'data/twitterdata/twitter-test.txt'
    trainrefer = 'data/twitterdata/twitter-refer.txt'
    train_object = open(trainpath, 'w')
    train_object_refer = open(trainrefer, 'w')
    text2id = {}
    tests = []
    refers = []
    with open(filename) as f:
        for line in f:
            text = line.split('\t')
            if len(text)==4:
		        pass
            else:
                continue
            tup = text[2]
            if int(tup[1])<= int(tup[3])/2:
		        continue
            s = utils.clarify(text[0]).lower().strip().strip('.')
            s1 = utils.clarify(text[1]).lower().strip().strip('.')
            if len(s.split())>15:
                temp = s
                s=s1
                s1=temp
                if len(s.split())>15:
                    continue
            tests.append(s)
            refers.append(s1)

    for x,y in zip(tests, refers):
        train_object.write(x)
        train_object.write('\n')
        train_object_refer.write(y)
        train_object_refer.write('\n')

    train_object.close()
    train_object_refer.close()


def generate_twitter_test(filename):
    trainpath = 'data/twitterdata/twitter-test.txt'
    trainrefer = 'data/twitterdata/twitter-refer.txt'
    train_object = open(trainpath, 'w')
    train_object_refer = open(trainrefer, 'w')
    text2id = {}
    tests = []
    refers = []
    test2refer = {}
    with open(filename) as f:
        for line in f:
            text = line.split('\t')
            if len(text)==4:
		        pass
            else:
                continue
            tup = text[2]
            if int(tup[1])<= int(tup[3])/2:
		        continue
            s = utils.clarify(text[0]).lower().strip().strip('.')
            s1 = utils.clarify(text[1]).lower().strip().strip('.')
            s1 = s1.replace('#','  ')
            s = ' '.join(s.split()[:15])
            if test2refer.has_key(s):
                test2refer[s].append(s1)
            else:
                test2refer[s] = s1

    for x,y in test2refer.items():
        train_object.write(x)
        train_object.write('\n')
        train_object_refer.write(' # '.join(y))
        train_object_refer.write('\n')

    train_object.close()
    train_object_refer.close()


def transfer2utf(filename):
    path = 'temp.txt'
    train_object = open(path, 'w')
    with open(filename) as f:
        for line in f:
            textsss = ''.join([x for x in line if (ord(x)<128 and ord(x)>-1)])
            textsss = textsss.decode('utf-8').encode('gb2312')
            train_object.write(textsss)
            # train_object.write('\n')
    train_object.close()


if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)
    transfer2utf(filename)
    # reproducefile(filename)


