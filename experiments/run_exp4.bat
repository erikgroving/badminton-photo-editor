@echo off
cd /d "C:\Users\erikg\OneDrive\Desktop\CodePlayground\badminton-photo-editor"
set PYTHONIOENCODING=utf-8
python experiments/exp4_spatial_features.py > logs\exp4_stdout.log 2>&1
echo Exit code: %ERRORLEVEL% >> logs\exp4_stdout.log
