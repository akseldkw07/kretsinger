import os
import subprocess


def source_zsh_env():
    # Run a zsh shell, source the profile, and print env as key=value
    command = ["zsh", "-c", "source ~/.zshrc >/dev/null 2>&1; env"]
    result = subprocess.run(command, capture_output=True, text=True)
    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    os.environ.update(env_vars)
    return env_vars
