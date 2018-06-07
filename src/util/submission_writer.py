import csv

def write_submission(results, outputFile='./../submission.csv'):

    with open(outputFile, 'w') as outcsv:
        writer = csv.writer(outcsv)
        for result in results:
            writer.writerow([result+1])
