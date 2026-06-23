import fire, time, os, h5py
from pathlib import Path
from tqdm import tqdm
import pickle

def create_reading_index(data_dir: Path, flow_inside_check=False):
    pkl_file_name = 'index_total.pkl' if not flow_inside_check else 'index_flow.pkl'
    start_time = time.time()
    data_index = []
    # os.listdir(data_dir)
    sorted_files = sorted(os.listdir(data_dir))
    for file_name in tqdm(sorted_files, ncols=100):
        if not file_name.endswith(".h5"):
            continue
        scene_id = file_name.split(".")[0]
        timestamps = []
        try:
            with h5py.File(data_dir/file_name, 'r') as f:
                if flow_inside_check:
                    for key in f.keys():
                        if 'flow' in f[key]:
                            # print(f"Found flow in {scene_id} at {key}")
                            timestamps.append(key)
                else:
                    timestamps.extend(f.keys())

        except Exception as e:
            print(f"Error reading {file_name}: {e}. Please try again with deleting this file.")
            break

        timestamps.sort(key=lambda x: int(x)) # make sure the timestamps are in order
        for timestamp in timestamps:
            data_index.append([scene_id, timestamp])

    with open(data_dir/pkl_file_name, 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create {pkl_file_name} index Successfully with {len(data_index)} frames, cost: {time.time() - start_time:.2f} s")

def create_eval_index(
    data_dir: str = "/home/kin/data/CARLA/demo/val",
    interval: int = 5,
):
    with open(os.path.join(data_dir, "index_total.pkl"), 'rb') as f:
        total_index = pickle.load(f)

    scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
    for idx, (scene_id, timestamp) in enumerate(total_index):
        if scene_id not in scene_id_bounds:
            scene_id_bounds[scene_id] = {
                "min_timestamp": timestamp, "max_timestamp": timestamp,
                "min_index": idx, "max_index": idx
            }
        else:
            bounds = scene_id_bounds[scene_id]
            if timestamp < bounds["min_timestamp"]:
                bounds["min_timestamp"] = timestamp
                bounds["min_index"] = idx
            if timestamp > bounds["max_timestamp"]:
                bounds["max_timestamp"] = timestamp
                bounds["max_index"] = idx

    # split the index by 5 - 5 frame, start with the fivth frame
    eval_data_index = []
    for scene_id, bounds in scene_id_bounds.items():
        with h5py.File(os.path.join(data_dir, f'{scene_id}.h5'), 'r') as f:
            for idx in range(bounds["min_index"] + interval*2, bounds["max_index"] - interval*2, interval):
                scene_id, timestamp = total_index[idx]
                key = str(timestamp)
                pc = f[key]['lidar'][:][:,:3]
                gm = f[key]['ground_mask'][:]
                if pc[~gm].shape[0] < 10000:
                    continue
                eval_data_index.append(total_index[idx])
    
    # print(f"Demo: {eval_data_index[:10]}")
    print(f"Total {len(eval_data_index)} frames for evaluation in {data_dir}.")
    with open(os.path.join(data_dir, "index_eval.pkl"), 'wb') as f:
        pickle.dump(eval_data_index, f)

def main(
    data_dir: str ="/home/kin/data/CARLA/data",
):
    create_reading_index(Path(data_dir))

if __name__ == "__main__":
    fire.Fire(main)
    # fire.Fire(create_eval_index)