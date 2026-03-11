# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils


def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    predict = open(
        "E:/projects/AIsolution/work6/a4/student/vanilla.pretrain.test.predictions", "r", encoding="utf-8")
    ans = 'London'
    for line in predict:
        if line.strip() == ans:
            accuracy += 1
    ### END YOUR CODE ###

    return accuracy


if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
