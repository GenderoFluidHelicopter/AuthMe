import os

class Cleaner():
    def __init__(self):
            self.folders_to_clear = ["./embeddings", "./dataset", "./set", "./cropped", "./prepared", "./processed_text", "./segmented", "./rotated"]
    def delete_folder_contents(self):
        try:
            for folder in self.folders_to_clear:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                os.rmdir(file_path)
                        except Exception as e:
                            print(f"Failed to delete {file_path}. Reason: {e}")
                    print(f"Contents of folder {folder} deleted successfully.")
                else:
                    print(f"Folder {folder} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")