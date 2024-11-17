def parse_img_classify():    
    import argparse
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='This is a simple command-line parser example.')

    # Add arguments
    parser.add_argument('-m', '--model', type=str, help='model name')
    parser.add_argument('-f', '--file', type=str, help='file path')

    # Parse the arguments
    return parser.parse_args()

