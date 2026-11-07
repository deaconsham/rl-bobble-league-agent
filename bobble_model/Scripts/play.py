"""
Play against the AI.

Builds the game the first time it runs, opens it in human vs AI mode, and plays
the AI's side with Models/best_final.pt. You are Team A (blue): left-drag a
player to aim (a longer drag is a harder shot), right-click to end your turn.
Close the game window to quit.

usage: python bobble_model/Scripts/play.py [--godot PATH] [--checkpoint PATH]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import evaluate
from launch_games import default_godot_exe, default_project_path
from shared_memory_link import Shared_memory_link


def is_built(project):
    """
    Docstring for is_built:

    True once Godot has imported the assets and the C# has been compiled. A fresh
    clone has neither, because both live in the gitignored .godot/ folder.
    """
    imported = os.path.isdir(os.path.join(project, '.godot', 'imported'))
    dll = os.path.join(project, '.godot', 'mono', 'temp', 'bin', 'Debug', 'BobbleLeague.dll')
    return imported and os.path.exists(dll)


def build(godot, project):
    """
    Docstring for build:

    Compile the C# scripts with dotnet, then let Godot import the assets once.
    """
    print('first run: building the game (takes a minute or two)...')
    try:
        subprocess.run(['dotnet', 'build', os.path.join(project, 'BobbleLeague.csproj')], check=True)
    except FileNotFoundError:
        sys.exit('dotnet not found. Install the .NET 8 SDK: https://dotnet.microsoft.com/download')
    subprocess.run([godot, '--headless', '--path', project, '--import'], check=False)


def wait_for_game(shm, game, timeout=120.0):
    """
    Docstring for wait_for_game:

    Wait until the game has opened its shared memory file, so the AI can talk to it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if game.poll() is not None:
            sys.exit('the game closed before it started. run it from the Godot editor to see why.')
        if os.path.exists(shm):
            try:
                Shared_memory_link(path=shm).close()
                return
            except (RuntimeError, ValueError):
                pass
        time.sleep(0.2)
    sys.exit(f'the game did not start within {timeout:.0f}s')


def quit_when_closed(game):
    """
    Docstring for quit_when_closed:

    The AI waits forever for your turn, so stop the whole script once the game window closes.
    """
    game.wait()
    print('game closed, bye')
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--godot', default=None,
                        help='Godot 4.6 .NET exe (default: $BOBBLE_GODOT_EXE, else godot on the PATH)')
    parser.add_argument('--checkpoint', default=None, help='model to play against (default: Models/best_final.pt)')
    args = parser.parse_args()

    godot = args.godot or default_godot_exe()
    if not os.path.exists(godot) and not shutil.which(godot):
        sys.exit(f"Godot not found at '{godot}'. Pass --godot or set BOBBLE_GODOT_EXE "
                 '(you need the .NET build of Godot 4.6).')
    project = default_project_path()
    if not is_built(project):
        build(godot, project)

    # the game and the AI find each other through this file
    shm = os.path.join(tempfile.gettempdir(), 'bobble_league_play.dat')
    if os.path.exists(shm):
        os.remove(shm)
    game = subprocess.Popen([godot, '--path', project, '--', '--mode=human_vs_ai', f'--shm={shm}'])
    wait_for_game(shm, game)
    threading.Thread(target=quit_when_closed, args=(game,), daemon=True).start()

    os.environ['BOBBLE_SHM_PATH'] = shm
    eval_args = ['--human', '--episodes', '0']
    if args.checkpoint:
        eval_args += ['--checkpoint', args.checkpoint]
    try:
        evaluate.main(eval_args)
    finally:
        if game.poll() is None:
            game.terminate()


if __name__ == '__main__':
    main()
