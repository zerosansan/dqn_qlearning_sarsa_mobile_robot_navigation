import os
import csv
import pickle


def record_data(data, outdir, file_name):
    """
        Saves training/testing data into a CSV file format.
    """
    file_exists = os.path.isfile(outdir + "/" + file_name + ".csv")
    with open(outdir + "/" + file_name + ".csv", "w") as fp:
        headers = ['episode_number', 'success_episode', 'failure_episode', 'episode_reward', 'episode_step']
        writer = csv.DictWriter(fp, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        wr = csv.writer(fp, dialect='excel')
        wr.writerow(data)


def load_q(file):
    """
        Loads trained Q-tables for Q-Learning or Sarsa algorithm.
    """
    q = None
    with open(file, 'rb') as f:
        q = pickle.loads(f.read())

    return q


def get_q(q, state, action):
    """
        Gets Q-values from the Q-table of Q-Learning or Sarsa algorithm.
    """
    return q.get((state, action), 0.0)
