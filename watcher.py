import os
import subprocess
import json

PIPE_IN = "/data/tygr_in"
PIPE_OUT = "/data/tygr_out"
OUTPUT_PKL = "/data/result.pkl"
TYGR_PYTHON = "/opt/conda/envs/TYGR/bin/python"

OPT_LEVELS = ["O0", "O1", "O2", "O3"]
ARCHS = ["aarch64", "arm32", "mips", "x64", "x86"]


def get_model(arch, opt_level):
    if opt_level not in OPT_LEVELS or arch not in ARCHS:
        raise ValueError(f"Invalid arch or opt_level: {arch}, {opt_level}")
    return f"model/MODEL_base/{arch}.{opt_level}.base.model"


if not os.path.exists(PIPE_IN):
    os.mkfifo(PIPE_IN)
if not os.path.exists(PIPE_OUT):
    os.mkfifo(PIPE_OUT)

print("WATCHER: Listening on Named Pipes...")

while True:
    try:
        with open(PIPE_IN, "r") as p_in:
            raw_msg = p_in.read().strip()

        if not raw_msg:
            continue

        req = json.loads(raw_msg)
        target_file = os.path.join("/data", req["file_path"])
        arch = req["arch"]
        opt_level = req["opt_level"]

        model = get_model(arch, opt_level)

        print(f"Processing: {target_file} with optimization {opt_level}")

        cmd = [
            TYGR_PYTHON,
            "-m",
            "src.index",
            "predict",
            model,
            target_file,
            OUTPUT_PKL,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        with open(PIPE_OUT, "w") as p_out:
            if result.returncode == 0:
                p_out.write("SUCCESS")
            else:
                p_out.write(f"ERROR: {result.stderr}")

    except json.JSONDecodeError:
        with open(PIPE_OUT, "w") as p_out:
            p_out.write("ERROR: Invalid JSON!")

    except Exception as e:
        with open(PIPE_OUT, "w") as p_out:
            p_out.write(f"SYSTEM_ERROR: {str(e)}")
