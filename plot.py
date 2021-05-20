import matplotlib.pyplot as plt
import csv
import os


def plot_history(location):
    with open(location + "/plot.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        got_header = False
        for row in csv_reader:
            if not got_header:
                headers = row
                data = []
                for i in range(0, len(headers)):
                    data.append([])
                got_header = True
            else:
                for i in range(0, len(headers)):
                    data[i].append(float(row[i]))

    for i in range(1, len(headers)):
        plt.plot(data[0], data[i])
        plt.xlabel(headers[0])
        plt.ylabel(headers[i])

        current_dir = os.getcwd()
        plt.savefig(current_dir + "\\" + location.replace("/", "\\") + '\\{}-{}.png'.format(headers[0], headers[i]).replace(" ", "_"))

        plt.clf()


