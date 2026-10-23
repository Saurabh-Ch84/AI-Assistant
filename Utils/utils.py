import os 

START_SEQUENCE="\033[90m" # ANSI escape code for grey text
END_SEQUENCE="\033[0m" # ANSI escape code to reset text color

def log(text):
    if os.getenv('DEBUG')=="TRUE":
        print(f"  {START_SEQUENCE}{text}{END_SEQUENCE}")

        