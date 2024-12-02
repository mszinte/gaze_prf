from gaze_prf.utils.data import get_all_subject_ids
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import product
from psc_preproc import main as filter_and_clean

def process_task(args):
    """Wrapper function to process a single (subject, session, task, run)."""
    subject, session, task, run, bids_folder = args
    try:
        filter_and_clean(subject, session, task, run, bids_folder)
    except Exception as e:
        print(f'Failed for sub-{subject} ses-{session} task-{task} run-{run}')
        print(e)

def main(bids_folder='/tank/shared/2021/visual/pRFgazeMod/'):
    subject_ids = get_all_subject_ids()
    sessions = [1, 2]
    tasks = ['AttendStimGazeCenter', 'AttendFixGazeCenter', 'AttendStimGazeLeft', 'AttendFixGazeLeft', 'AttendStimGazeRight', 'AttendFixGazeRight']
    tasks += ['AttendStimGazeCenterFS', 'AttendFixGazeCenterFS']

    # Generate all combinations of subject, session, task, and runs
    task_combinations = []
    for subject, session, task in product(subject_ids, sessions, tasks):
        runs = [1, 2] if task.endswith('FS') else [1]
        for run in runs:
            task_combinations.append((subject, session, task, run, bids_folder))

    # Use ProcessPoolExecutor with tqdm for parallel processing
    process_map(process_task, task_combinations, max_workers=16, desc="Processing tasks")  # Adjust max_workers as needed

if __name__ == '__main__':
    main()
