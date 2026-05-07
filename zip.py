import zipfile
import os

folder_path = "/Users/otisvaliant/Documents/Seyit/A2C/logs/dqn_roundabout"
output_name = "dqn_roundabout.zip"
dest_dir = "/Users/otisvaliant/Documents/Seyit/A2C/logs/dqn_roundabout/zip"


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

if __name__ == "__main__":
    zip_folder(folder_path, output_name, dest_dir)