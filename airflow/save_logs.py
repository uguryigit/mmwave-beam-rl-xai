import zipfile
import os
import minio
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter

def zip_folder(folder_path, output_name, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    output_path = os.path.join(dest_dir, output_name)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Skip the output zip itself if it's inside the folder
                if os.path.abspath(file_path) == os.path.abspath(output_path):
                    continue

                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

    print(f"✅ Zip created at: {output_path}")
    return output_path



def upload_to_minio(endpoint, access_key, secret_key, bucket_name, file_path, object_name=None):
    """Upload file to MinIO using minio client."""
    client = minio.Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # change to True if using https
    )
    client.fput_object(bucket_name, object_name, file_path)
    print(f"✅ Uploaded {file_path} to bucket '{bucket_name}' as '{object_name}'")


def main(folder, access_key, secret_key):
    LOG_ROOT = os.path.join(folder, "roundabout_q_factorized")
    LOG_ROOT2 = os.path.join(folder, "roundabout_dqn")
    # all_summaries = []
    # all_episode_rewards = []
    # total_wall_time_sec = 0
    # all_start_times = datetime.now().isoformat() + "Z"
    # all_end_times = datetime.now().isoformat() + "Z"
    # for root, dirs, files in os.walk(LOG_ROOT):
    #     for i, dir in enumerate(dirs):
    #         if dir.startswith("run_"):
    #             if (os.path.exists(os.path.join(root+"/"+dir, "run_summary.json"))):
    #                 with open(os.path.join(root+"/"+dir, "run_summary.json"), "r") as f:
    #                     summary = json.load(f)
    #                 if i == 0:
    #                     all_start_times = summary["start_time"]
    #                 if i == len(dirs)-1:
    #                     all_end_times = summary["end_time"]
    #                 all_summaries.append(summary)
    #                 total_wall_time_sec += summary["total_train_time_sec"]
    #             if (os.path.exists(os.path.join(root+"/"+dir, "runs.json"))):
    #                 with open(os.path.join(root+"/"+dir, "runs.json"), "r") as f:
    #                     episode_rewards = json.load(f)
    #                 all_episode_rewards.append({"run_id": int(dir.split("_")[1]), "episode_rewards": episode_rewards})
    
    # all_summaries = sorted(all_summaries, key=lambda x: x["run_id"], reverse=False)
    # all_episode_rewards = sorted(all_episode_rewards, key=lambda x: x["run_id"], reverse=False)
                    


    # global_json = os.path.join(LOG_ROOT, "all_runs_summary.json")
    # with open(global_json, "w") as fjs:
    #     json.dump({
    #         "total_runs": len(all_summaries),
    #         "total_wall_time_sec": total_wall_time_sec,
    #         "start_time": all_start_times,
    #         "end_time": all_end_times,
    #         "runs": all_summaries
    #     }, fjs, indent=2)

    # if SummaryWriter:
    #     GLOBAL_LOG_DIR = os.path.join(LOG_ROOT, "_global")  # separate TB run for aggregates
        
    #     # 1) Split all runs into 3 segments and find the best in each segment
    #     total_runs = len(all_summaries)
    #     segment_size = total_runs // 3
    #     remainder = total_runs % 3
        
    #     # Calculate segment boundaries
    #     segments = []
    #     start_idx = 0
    #     for i in range(3):
    #         # Distribute remainder across first segments
    #         current_size = segment_size + (1 if i < remainder else 0)
    #         end_idx = start_idx + current_size
    #         segments.append((start_idx, end_idx))
    #         start_idx = end_idx
        
    #     # Find best run in each segment
    #     for seg_idx, (start, end) in enumerate(segments):
    #         segment_runs = all_summaries[start:end]
    #         if segment_runs:
    #             best_in_segment = max(segment_runs, key=lambda s: s["last_window_avg_reward"])
    #             # Log the best run from this segment
    #             best_run_id = best_in_segment["run_id"]
    #             best_episode_rewards = all_episode_rewards[best_run_id]["episode_rewards"]
    #             global_writer = SummaryWriter(GLOBAL_LOG_DIR+"/Best_Segment_"+str(seg_idx)+"_Run_"+str(best_run_id))
    #             for episode, reward in best_episode_rewards:
    #                 global_writer.add_scalar(
    #                     f"Global/Segments/Best/episode_reward",
    #                     reward,
    #                     global_step=episode,
    #                 )
    #     global_writer.flush()
    #     global_writer.close()



    #     all_summaries = []
    
    
    
    # DQN
    all_episode_rewards = []
    total_wall_time_sec = 0
    all_start_times = datetime.now().isoformat() + "Z"
    all_end_times = datetime.now().isoformat() + "Z"
    for root, dirs, files in os.walk(LOG_ROOT2):
        for i, dir in enumerate(dirs):
            if dir.startswith("run_"):
                if (os.path.exists(os.path.join(root+"/"+dir, "run_summary.json"))):
                    with open(os.path.join(root+"/"+dir, "run_summary.json"), "r") as f:
                        summary = json.load(f)
                    if i == 0:
                        all_start_times = summary["start_time"]
                    if i == len(dirs)-1:
                        all_end_times = summary["end_time"]
                    all_summaries.append(summary)
                    total_wall_time_sec += summary["train_wall_time_sec"]
                if (os.path.exists(os.path.join(root+"/"+dir, "runs.json"))):
                    with open(os.path.join(root+"/"+dir, "runs.json"), "r") as f:
                        episode_rewards = json.load(f)
                    all_episode_rewards.append({"run_id": int(dir.split("_")[1]), "episode_rewards": episode_rewards})
    
    all_summaries = sorted(all_summaries, key=lambda x: x["run_id"], reverse=False)
    all_episode_rewards = sorted(all_episode_rewards, key=lambda x: x["run_id"], reverse=False)
                    


    global_json = os.path.join(LOG_ROOT2, "all_runs_summary.json")
    with open(global_json, "w") as fjs:
        json.dump({
            "total_runs": len(all_summaries),
            "total_wall_time_sec": total_wall_time_sec,
            "start_time": all_start_times,
            "end_time": all_end_times,
            "runs": all_summaries
        }, fjs, indent=2)

    if SummaryWriter:
        GLOBAL_LOG_DIR = os.path.join(LOG_ROOT2, "_global")  # separate TB run for aggregates
        
        # 1) Split all runs into 3 segments and find the best in each segment
        total_runs = len(all_summaries)
        segment_size = total_runs // 3
        remainder = total_runs % 3
        
        # Calculate segment boundaries
        segments = []
        start_idx = 0
        for i in range(3):
            # Distribute remainder across first segments
            current_size = segment_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            segments.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Find best run in each segment
        for seg_idx, (start, end) in enumerate(segments):
            segment_runs = all_summaries[start:end]
            if segment_runs:
                best_in_segment = max(segment_runs, key=lambda s: s["final_eval_mean_reward"])
                # Log the best run from this segment
                best_run_id = best_in_segment["run_id"]
                best_episode_rewards = all_episode_rewards[best_run_id]["episode_rewards"]
                global_writer = SummaryWriter(GLOBAL_LOG_DIR+"/Best_Segment_"+str(seg_idx)+"_Run_"+str(best_run_id))
                for episode, reward in best_episode_rewards:
                    global_writer.add_scalar(
                        f"Global/Segments/Best/episode_reward",
                        reward,
                        global_step=episode,
                    )
        global_writer.flush()
        global_writer.close()


    zip_file = datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".zip"
    # Step 1: Create zip
    output = zip_folder(folder, zip_file, folder+"/zip" )

    # Step 2: Upload to MinIO
    upload_to_minio(
        endpoint="minio-hl.str.svc.cluster.local:9000",   # Your MinIO server
        access_key=access_key,
        secret_key=secret_key,
        bucket_name="reinforcement",
        file_path=output,
        object_name=zip_file
    )


