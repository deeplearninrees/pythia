import subprocess
import os
import time
#I know I'm downloading using a subprocess call when this can be done using the Python library... 
#Just that the CLI is more documented.


def main_downloader():
    with open("../links.txt") as f:
        iter_links = f.readlines()
        i = 0
        if os.path.exists("./linktostartfrom.txt"):
            i = int(open("./linktostartfrom.txt", "r").read())-25
        while i < len(iter_links):
            line = iter_links[i]
            output_path = os.path.join('../videos/', line[-12:]+'.mp4')
            if not os.path.exists(output_path):
                process = subprocess.call(['../yt-dlp',
                                "-f", "best", 
                                "--merge-output-format", "mp4", 
                                "--match-filter", '!is_live & duration < 2050',
                                "-o", output_path,
                                "--throttled-rate", "5M",
                                line, 
                                ])
                open('./linktostartfrom.txt', 'w').write(str(i))
            i += 1

if __name__ == "__main__":
    main_downloader()