{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/bin/zsh\
\
# This script activates a specific Conda environment and navigates to a project directory.\
# It should be SOURCED, not executed directly.\
# Example: source ~/bin/setup_my_env.zsh\
\
# --- User Configuration ---\
# 1. Name of your Conda environment\
ENV_NAME="kret_312" \
\
# 2. Path to your project directory\
TARGET_DIR=\'93/Users/Akseldkw/coding/kretsinger\'93 \
\
# 3. (Optional) Define any specific shell commands, aliases, or functions\
#    that you want to be available *only* when this environment is active\
#    and you are in this directory.\
\
# Example:\
# ALIAS_NAME="run_dev_server"\
# ALIAS_COMMAND="python app.py --env dev"\
# alias $\{ALIAS_NAME\}="$\{ALIAS_COMMAND\}"\
# echo "Alias '$\{ALIAS_NAME\}' created to run: $\{ALIAS_COMMAND\}"\
# -------------------------\
\
echo "--- Setting up Environment for $\{ENV_NAME\} ---"\
\
# Check if conda command is available\
if ! command -v conda &> /dev/null; then\
    echo "Error: 'conda' command not found. Please ensure Conda is installed and initialized for your shell."\
    return 1 # 'return' is used in sourced scripts instead of 'exit'\
fi\
\
# Check if the Conda environment exists\
if ! conda env list | grep -q "$\{ENV_NAME\}"; then\
    echo "Error: Conda environment '$\{ENV_NAME\}' not found. Please create it first."\
    return 1\
fi\
\
# Activate the Conda environment\
# Using 'conda activate' directly modifies the current shell's environment.\
echo "Activating Conda environment: $\{ENV_NAME\}"\
if ! conda activate "$\{ENV_NAME\}"; then\
    echo "Failed to activate '$\{ENV_NAME\}'. Check Conda installation and environment name."\
    return 1\
fi\
\
# Navigate to the specified location\
if [ -d "$\{TARGET_DIR\}" ]; then\
    echo "Navigating to: $\{TARGET_DIR\}"\
    # Use 'cd -P' to resolve symlinks and ensure canonical path\
    if ! cd -P "$\{TARGET_DIR\}"; then\
        echo "Failed to change directory to '$\{TARGET_DIR\}'. Check path permissions."\
        return 1\
    fi\
else\
    echo "Error: Target directory '$\{TARGET_DIR\}' does not exist. Please create it or update the script."\
    return 1\
fi\
\
# Source the .zshrc file at the very end.\
# WARNING: This is generally NOT necessary and can sometimes lead to redundant PATH entries\
# or unexpected behavior if your .zshrc contains complex logic or adds to PATH without\
# checking for existing entries. Your .zshrc is typically sourced automatically when\
# your shell starts. Conda activation usually handles necessary PATH modifications.\
# Only include this if you have specific reasons (e.g., dynamic updates in .zshrc\
# that rely on the environment being fully active).\
echo "Sourcing ~/.zshrc to ensure latest shell configurations..."\
if [ -f "$\{HOME\}/.zshrc" ]; then\
    source "$\{HOME\}/.zshrc"\
else\
    echo "Warning: ~/.zshrc not found. Skipping sourcing."\
\
echo "Environment setup complete! You are now in the '$\{ENV_NAME\}' environment at '$\{PWD\}'."\
}