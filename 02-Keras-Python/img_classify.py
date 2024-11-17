# usage: python img_classify.py -m resnet50 -f haku.png
from classifiers import resnet50_classify, xception_classify
from parser import parse_img_classify

# check if main
if __name__ == "__main__":
    args = parse_img_classify()

    if args.model == "resnet50":
        decoded_preds = resnet50_classify(args.file, top=3)
    elif args.model == "xception":
        decoded_preds = xception_classify(args.file, top=3)
    else:
        raise Exception("Invalid model")

    print('Predicted:', decoded_preds)