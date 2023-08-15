
import sys
import inspect
import os

caller = sys._getframe(1)
print(caller.f_locals)

abs_path = os.path.abspath(inspect.stack()[1][1])
directory_of_1py = os.path.dirname(abs_path)
path=os.path.join(directory_of_1py, "tests/active_sarsa.json")

with open(path) as f:
    print(f.read())