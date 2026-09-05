@echo off
REM Session-detached launcher for the compute-only tree-baseline queue.
REM Ruling of 2026-09-05: managed background tasks are killed at a hard ~60 min
REM cap, so the queue is started once via schtasks instead.  Same interpreter,
REM same config, same logs, same lock as the managed launch -- this file only
REM adds console redirection, because a scheduled task has no console.
REM ASCII only on purpose: cmd reads this file in the OEM code page.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\quant\us-quant-pipeline\scripts\run_gbdt_sealed_queue.ps1" >> "F:\quant\us-quant-pipeline\outputs\gbdt_strong_jkp_v2\xgboost\sealed\_logs\queue_task.log" 2>&1
