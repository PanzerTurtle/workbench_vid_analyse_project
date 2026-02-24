import sys
from tkinter import filedialog


def select_file(INITIAL_DIR="./"):
    INPUT_VIDEO = filedialog.askopenfilename(initialdir=INITIAL_DIR, title="Select video", filetypes=[("Video", "*.mp4")])

    if INPUT_VIDEO:
        print(f"Input path: \"{INPUT_VIDEO}\"")
    else:
        print("No video selected")
        sys.exit(0)
        
    return INPUT_VIDEO

if __name__ == "__main__":
    select_file()
