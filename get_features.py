import argparse
from saves import save_classes, save_classes_tensor
from img_features import get_image_features
from text_features import text_features, spacy_features

def main():
    parser = argparse.ArgumentParser(
        description="Get Visual and Textual features for flickr30k dataset"
    )

    parser.add_argument(
        '--visual', dest='visual',
        help="Use this flag to get visual features",
        action='store_true'
    )

    parser.add_argument(
        '--textual', dest='textual',
        help="Use this flag to get textual features",
        action='store_true'
    )


    args = parser.parse_args()

    if args.visual:
        save_classes()
        save_classes_tensor()
        get_image_features()

    if args.textual:
        spacy_features()
        text_features()

    return None

if '__name__' == '__main__':
    main()