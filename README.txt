installation:
pip install gym-retro

Copy SuperStreetFighter2-Snes-main into PYTHON_DIR\Lib\site-packages\retro\data\stable (base of this from https://github.com/SanjoSolutions/SuperStreetFighter2-Snes)

run
python3 -m retro.import /path/to/your/ROMs/directory/
to import the SSF2 rom included



QLearningAgent.py is the file that carries out training and a (usually)
single test run with epsilon = 1.

SSF2Discretizer.py defines our action/button space.

runFromLoadedQTable.py will run the game on a specific
policy given the path to it (these are saved for each
attempt in QLearningAgent.py).

