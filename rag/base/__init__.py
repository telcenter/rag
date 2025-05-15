import pandas as pd # type: ignore

LOG_FILENAME = "rag.log"
def log_info(message: str):
    # with open(LOG_FILENAME, "a") as f:
    #     f.write(f"{pd.Timestamp.now()}: {message}\n")
    print(message)

def log_info_and_print(message: str):
    # log_info(message)
    # print(message)
    pass
