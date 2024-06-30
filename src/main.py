import argparse
import logging


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pipeline for the thesis work "Mentoring Deep Learning Models for Mass Screening with Limited Data')
    parser.add_argument('-m','--model', type=str, nargs='?', required=True, default='vgg16', help= "Select model to obtain the deep features")
    parser.add_argument('-d','--distance', type=str, nargs='?', required=True, default='euclidean', help = "Select distance" )
    parser.add_argument('-s',"--subcluster", type=bool, nargs='?', required=True, default=False)
    parser.add_argument('-nos',"--nosub", type=int, nargs='?', default=5)

    args=parser.parse_args()
    print(args)