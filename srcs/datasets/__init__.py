import inspect
import os
import sys

current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dirlevel2 = os.path.dirname(current_path)
dirlevel1 = os.path.dirname(dirlevel2)
dirlevel0 = os.path.dirname(dirlevel1)

sys.path.insert(0, dirlevel0)
