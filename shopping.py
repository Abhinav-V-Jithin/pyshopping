import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename): # successfully implemented
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []

    with open(filename, encoding="utf-8") as csvfile:
        full_data = csv.reader(csvfile)
        for user_session in full_data:
            user_evidence = []


            for index in range(0, 18):

                data = user_session[index]
                try:
                    if index == 0 or index == 2 or index == 4 or index == 11 or index == 12 or index == 13 or index == 14:
                        data = int(user_session[index])
                    
                    elif index == 1 or index == 3 or index == 5 or index == 6 or index == 7 or index == 8 or index ==9:
                        data = float(user_session[index])

                    elif index == 10:
                        month = user_session[index]

                        month_number = {
                            "Jan" : 0,
                            "Feb" : 1,
                            "Mar" : 2,
                            "Apr" : 3,
                            "May" : 4,
                            "June" : 5,
                            "Jul" : 6,
                            "Aug" : 7,
                            "Sep" : 8,
                            "Oct" : 9,
                            "Nov" : 10,
                            "Dec" : 11,
                        }

                        data = month_number[month]
                    elif index == 15:
                        if data == "Returning_Visitor":
                            data = 1
                        else:
                            data = 0
                    elif index == 16:
                        if data == "FALSE":
                            data = 0
                        else:
                            data = 1

 
                except:
                    pass
                if index == 17:
                    if data == "FALSE":
                        data = 0
                    else:
                        data = 1

                    labels.append(data)
                    continue
                user_evidence.append(data)
 
            evidence.append(user_evidence)

    evidence.pop(0)
    labels.pop(0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)

    evidence_training = [row for row in evidence]
    labels_training = [row for row in labels]

    model.fit(evidence_training, labels_training)

    return model


def evaluate(labels, predictions): # success
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    tot = 0
    true_positive_count = 0
    actual_positive_count = 0
    true_negative_count = 0
    actual_negative_count = 0

    for label, prediction in zip(labels, predictions):
        tot += 1
        if label == 1:
            actual_positive_count += 1
            if label == prediction:
                true_positive_count += 1
        else:
            actual_negative_count += 1
            if label == prediction:
                true_negative_count += 1

    sensitivity = true_positive_count * 1.0 / actual_positive_count
    specificity = true_negative_count * 1.0 / actual_negative_count
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
