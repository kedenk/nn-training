@ECHO OFF 
if [%1]==[] goto usage

docker run -it --rm -v %cd%/%1:/tmp -w /tmp tensorflow/tensorflow python ./main.py

goto :eof
:usage
@echo Usage: %0 ^<directoryName^>
exit /B 1