import os
import csv


# Logging methods
def remove_logfile_if_exist(outdir, gazebo_world_launch_name):
    """
        Removes CSV logfile if exist.
    """
    try:
        os.remove(outdir + "/" + gazebo_world_launch_name + ".csv")
    except OSError:
        pass


def remove_qfile_if_exist(outdir, file):
    """
        Removes Q-table file if exist.
    """
    try:
        os.remove(outdir + "/" + file + ".txt")
    except OSError:
        pass


def record_data(data, outdir, gazebo_world_launch_name):
    """
        Saves training/testing data into a CSV file format.
    """
    file_exists = os.path.isfile(outdir + "/" + gazebo_world_launch_name + ".csv")
    with open(outdir + "/" + gazebo_world_launch_name + ".csv", "a") as fp:
        headers = ['episode_number', 'success_episode', 'failure_episode', 'episode_reward', 'episode_step']
        writer = csv.DictWriter(fp, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        wr = csv.writer(fp, dialect='excel')
        wr.writerow(data)
