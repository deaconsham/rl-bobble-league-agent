"""
Launch N headless Godot games, each with its own shared-memory file and log.

Each game i gets:
  --shm=<temp>/bobble_league_mmap_<i>.dat   so the files never clash
  logs/godot_<i>.log                        stdout + stderr

The games are launched DETACHED. On Windows the agent's shell tool hangs if
Godot runs as its child, so we start each process through WMI
(Win32_Process.Create), which reparents it away from the caller. Elsewhere we
use subprocess with its own session.

The Vector_env uses Game_launcher to start the games and to relaunch a single
one after a crash. launch_games.py can also be run on its own to bring games up
by hand (then train with training_loop.py --games N --shm-prefix ...).
"""
import os
import platform
import subprocess
import sys
import tempfile

from shared_memory_link import Shared_memory_link


# the real game on Deacon's machine. override with --godot or the env var.
DEFAULT_GODOT = (
    r"C:\Users\Decon Sham\Desktop\Godot_v4.6.1-stable_mono_win64"
    r"\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64_console.exe"
)
GODOT_ENV_VAR = "BOBBLE_GODOT_EXE"
PROJECT_ENV_VAR = "BOBBLE_PROJECT_PATH"


def default_godot_exe():
    """Docstring for default_godot_exe: env var wins, else the known Desktop path."""
    return os.environ.get(GODOT_ENV_VAR) or DEFAULT_GODOT


def default_project_path():
    """Docstring for default_project_path: the bobble_league Godot project folder."""
    from_env = os.environ.get(PROJECT_ENV_VAR)
    if from_env:
        return from_env
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "bobble_league")


def shm_path_for(index, prefix=None):
    """
    Docstring for shm_path_for:

    Per-game shared-memory file path. Game 0 -> ..._mmap_0.dat, and so on, so the
    N games never share a buffer. prefix defaults to the system temp dir.
    """
    prefix = prefix or tempfile.gettempdir()
    return os.path.join(prefix, f"bobble_league_mmap_{index}.dat")


def log_path_for(index, log_dir):
    """Docstring for log_path_for: logs/godot_<i>.log under log_dir."""
    return os.path.join(log_dir, f"godot_{index}.log")


class Game_launcher:
    """
    Docstring for Game_launcher class:

    Starts and tracks N detached headless Godot processes, one per game. Keeps
    the per-game shm paths, log paths and pids so a crashed game can be relaunched
    on its own index without touching the others.

    Args:
        n_games: how many games to run.
        godot_exe: path to the Godot console binary (default: the Desktop build).
        project_path: the bobble_league project folder.
        shm_prefix: folder for the per-game shm files (default: system temp).
        log_dir: folder for godot_<i>.log (default: bobble_model/logs).
        timescale: AI_VS_AI time scale passed to each game.
    """

    def __init__(self, n_games, godot_exe=None, project_path=None, shm_prefix=None,
                 log_dir=None, timescale=48):
        self.n_games = n_games
        self.godot_exe = godot_exe or default_godot_exe()
        self.project_path = os.path.abspath(project_path or default_project_path())
        self.shm_prefix = os.path.abspath(shm_prefix or tempfile.gettempdir())
        self.log_dir = os.path.abspath(
            log_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
        self.timescale = timescale
        os.makedirs(self.log_dir, exist_ok=True)

        self.shm_paths = [shm_path_for(i, self.shm_prefix) for i in range(n_games)]
        self.log_paths = [log_path_for(i, self.log_dir) for i in range(n_games)]
        self.pids = [None] * n_games

    def _args(self, index):
        """Docstring for _args: the full Godot argument list for one game."""
        return [
            self.godot_exe,
            "--path", self.project_path,
            "--headless",
            "--",
            "--mode=ai_vs_ai",
            f"--timescale={self.timescale}",
            f"--shm={self.shm_paths[index]}",
        ]

    def launch_one(self, index):
        """
        Docstring for launch_one:

        Start (or restart) the game at this index, detached, logging to its file.
        Removes a stale shm file first so the version-byte check does not read an
        old buffer. Returns the pid (or None off Windows, where we cannot get it
        back from WMI cleanly but still track the subprocess).
        """
        shm = self.shm_paths[index]
        if os.path.exists(shm):
            try:
                os.remove(shm)
            except OSError:
                pass

        log = self.log_paths[index]
        if platform.system() == "Windows":
            pid = self._launch_windows(index, log)
        else:
            pid = self._launch_posix(index, log)
        self.pids[index] = pid
        return pid

    def _launch_windows(self, index, log):
        """
        Docstring for _launch_windows:

        Detached launch through WMI so the process is not a child of the shell
        tool (a child Godot hangs the agent shell). The command (quoted exe, args
        and a redirect to the log) is written to a per-game .bat so no quotes are
        mangled by the python -> powershell -> cmd nesting, then WMI runs that bat.
        """
        quoted = " ".join(f'"{a}"' if (" " in a) else a for a in self._args(index))
        bat = os.path.join(self.log_dir, f"run_godot_{index}.bat")
        with open(bat, "w") as f:
            f.write("@echo off\r\n")
            f.write(f'{quoted} > "{log}" 2>&1\r\n')

        ps = (
            "$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            f"-Arguments @{{CommandLine = 'cmd /c \"\"{bat}\"\"'}}; "
            "Write-Output $p.ProcessId"
        )
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            text=True,
        )
        pid = None
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
        return pid

    def _launch_posix(self, index, log):
        """Docstring for _launch_posix: detached launch in its own session."""
        with open(log, "wb") as f:
            proc = subprocess.Popen(
                self._args(index),
                stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return proc.pid

    def launch_all(self):
        """Docstring for launch_all: start every game."""
        for i in range(self.n_games):
            self.launch_one(i)
        return self.pids

    def wait_for_handshake(self, index, timeout=60.0):
        """
        Docstring for wait_for_handshake:

        Attach to the game's shm file once it exists and the version byte matches,
        so the caller knows the game booted. Returns a live Shared_memory_link.
        """
        import time
        shm = self.shm_paths[index]
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(shm):
                try:
                    return Shared_memory_link(path=shm, create=False)
                except (RuntimeError, ValueError):
                    # file exists but version byte not written yet; keep waiting
                    pass
            time.sleep(0.1)
        raise TimeoutError(f"Godot game {index} never handshook on '{shm}' within {timeout}s")

    def stop_one(self, index):
        """
        Docstring for stop_one:

        Kill only the Godot for this game. WMI's Win32_Process.Create returns the
        cmd wrapper pid, not Godot's (cmd spawns Godot then exits), so we find the
        real process by its unique --shm path on the command line and kill that.
        """
        if platform.system() == "Windows":
            self._kill_by_shm(self.shm_paths[index])
        elif self.pids[index] is not None:
            self._kill(self.pids[index])
        self.pids[index] = None

    def stop_all(self):
        """Docstring for stop_all: kill every game we started, and only those."""
        for i in range(self.n_games):
            self.stop_one(i)

    @staticmethod
    def _kill_by_shm(shm):
        """
        Docstring for _kill_by_shm:

        Kill any process whose command line carries this exact --shm path. Each
        game's path is unique, so this stops that one game and no other.
        """
        ps = (
            "Get-CimInstance Win32_Process "
            "| Where-Object { $_.Name -like 'Godot*' -and $_.CommandLine -like '*--shm=" + shm + "*' } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        except (OSError, subprocess.SubprocessError):
            pass

    @staticmethod
    def _kill(pid):
        """Docstring for _kill: platform kill by pid, errors swallowed."""
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, 9)
        except (OSError, subprocess.SubprocessError):
            pass


def main():
    """Docstring for main: bring up N games by hand for a manual training run."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch N headless Godot games (detached).")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--godot", default=None, help="path to the Godot console exe")
    parser.add_argument("--project", default=None, help="the bobble_league project folder")
    parser.add_argument("--shm-prefix", default=None, help="folder for the per-game shm files")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--timescale", type=int, default=48)
    args = parser.parse_args()

    launcher = Game_launcher(args.games, godot_exe=args.godot, project_path=args.project,
                             shm_prefix=args.shm_prefix, log_dir=args.log_dir,
                             timescale=args.timescale)
    pids = launcher.launch_all()
    print(f"launched {args.games} games, pids: {pids}")
    for i in range(args.games):
        launcher.wait_for_handshake(i)
        print(f"  game {i} up: shm={launcher.shm_paths[i]} log={launcher.log_paths[i]}")
    print("games are up. train with: python training_loop.py --games", args.games,
          "--shm-prefix", launcher.shm_prefix)


if __name__ == "__main__":
    main()
