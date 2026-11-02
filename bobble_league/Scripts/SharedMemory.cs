using Godot;
using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;

// bridge between godot and python for RL training.
// uses a FILE-BACKED memory map so it works on Windows, macOS and Linux
// (named maps without a file are Windows-only in .NET).
//
// Protocol v5 layout (124 bytes). Must match bobble_model/shared_memory_link.py.
//   0-55   obs, 14 x f32           godot -> python  (learner's frame, see DataManager)
//   56-79  learner action, 6 x f32 python -> godot  [a1,p1,a2,p2,a3,p3] angle+power, learner's frame
//   80-83  reward, f32             godot -> python  (learner's point of view)
//   84     done, u8                godot -> python  (1 = goal scored)
//   85     step request, u8        python sets 1, godot clears
//   86     reset request, u8       python sets 1, godot clears
//   87     learner team, u8        godot -> python  (1 or 2)
//   88     scorer, u8              godot -> python  (0 none, 1 learner, 2 opponent)
//   89     play mode, u8           godot -> python  (GameManager.PlayMode)
//   90     opponent action valid   python -> godot  (1 = use bytes 92-115)
//   91     random start, u8        python -> godot  (1 = scatter the next reset) (v5)
//   92-115 opponent action, 6 x f32 python -> godot (opponent's own frame, angle+power)
//   116    protocol version, u8    godot -> python
//   117    oob reset, u8           godot -> python  (1 = this turn hit an out-of-bounds kickoff) (v5)
//   118-119 pad
//   120-123 turn seconds, f32      godot -> python  (game seconds the last turn took to settle)
// v4 only GREW the buffer with turn-length telemetry; the action/obs contract is
// unchanged, so checkpoints keep their own action_format tag (still 3). v5 claims
// two pad bytes (Phase 12, step 1): byte 91 is a python->godot random-start flag
// and byte 117 a godot->python out-of-bounds flag. Both only steer bodies /
// telemetry, so the action/obs contract is unchanged and action_format stays 3.
public class SharedMemory : IDisposable {
    public const byte PROTOCOL_VERSION = 5;

    private const int OBS_OFFSET = 0;
    private const int ACT_OFFSET = 56;
    private const int REWARD_OFFSET = 80;
    private const int DONE_OFFSET = 84;
    private const int STEP_REQ_OFFSET = 85;
    private const int RESET_REQ_OFFSET = 86;
    private const int LEARNER_TEAM_OFFSET = 87;
    private const int SCORER_OFFSET = 88;
    private const int PLAY_MODE_OFFSET = 89;
    private const int OPP_VALID_OFFSET = 90;
    private const int RANDOM_START_OFFSET = 91;
    private const int OPP_ACT_OFFSET = 92;
    private const int VERSION_OFFSET = 116;
    private const int OOB_RESET_OFFSET = 117;
    private const int TURN_SECONDS_OFFSET = 120;
    private const int TOTAL_SIZE = 124;

    public const int OBS_DIM = 14;
    public const int ACT_DIM = 6;

    public const string DefaultFileName = "bobble_league_mmap.dat";
    public const string PathEnvVar = "BOBBLE_SHM_PATH";

    private FileStream _file;
    private MemoryMappedFile _mmf;
    private MemoryMappedViewAccessor _accessor;

    public string FilePath { get; private set; }
    public bool IsReady => _accessor != null;

    public SharedMemory() {
        InitializeSharedMemory();
    }

    // path rule: --shm=<path> on the command line wins, then the env var,
    // else <temp dir>/bobble_league_mmap.dat. the command-line option lets
    // launch_games.py give each parallel godot its own file.
    public static string ResolvePath() {
        string fromCmd = CmdlineShmPath();
        if (!string.IsNullOrEmpty(fromCmd)) {
            return fromCmd;
        }
        string fromEnv = System.Environment.GetEnvironmentVariable(PathEnvVar);
        return string.IsNullOrEmpty(fromEnv)
            ? Path.Combine(Path.GetTempPath(), DefaultFileName)
            : fromEnv;
    }

    // reads --shm=<path> from the args after "--", same place GameManager reads
    // --mode / --timescale. returns null when it is not given.
    static string CmdlineShmPath() {
        foreach (string arg in OS.GetCmdlineUserArgs()) {
            if (arg.StartsWith("--shm=")) {
                return arg.Substring("--shm=".Length);
            }
        }
        return null;
    }

    // creates (or reuses) the backing file, maps it and zeroes it
    private void InitializeSharedMemory() {
        try {
            FilePath = ResolvePath();

            _file = new FileStream(FilePath, FileMode.OpenOrCreate, System.IO.FileAccess.ReadWrite, FileShare.ReadWrite);
            if (_file.Length < TOTAL_SIZE) {
                _file.SetLength(TOTAL_SIZE);
            }

            // mapName must be null off Windows
            _mmf = MemoryMappedFile.CreateFromFile(
                _file, null, TOTAL_SIZE,
                MemoryMappedFileAccess.ReadWrite,
                HandleInheritability.None,
                leaveOpen: true);
            _accessor = _mmf.CreateViewAccessor(0, TOTAL_SIZE, MemoryMappedFileAccess.ReadWrite);

            // clear stale flags/actions from an old run
            for (int i = 0; i < TOTAL_SIZE; i++) {
                _accessor.Write(i, (byte)0);
            }
            _accessor.Write(VERSION_OFFSET, PROTOCOL_VERSION);
            _accessor.Flush();

            GD.Print($"Shared memory v{PROTOCOL_VERSION} ready at '{FilePath}' ({TOTAL_SIZE} bytes)");
        } catch (Exception e) {
            GD.PrintErr($"Failed to init shared memory at '{FilePath}': {e}");
            Dispose();
        }
    }

    float[] ReadFloats(int offset, int count) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
            values[i] = _accessor.ReadSingle(offset + i * 4);
        }
        return values;
    }

    public bool IsStepRequested() {
        return IsReady && _accessor.ReadByte(STEP_REQ_OFFSET) == 1;
    }

    public bool IsResetRequested() {
        return IsReady && _accessor.ReadByte(RESET_REQ_OFFSET) == 1;
    }

    // v5: python sets byte 91 before a reset request to ask for a scattered
    // (random legal layout) reset instead of kickoff. read it while handling the
    // reset. it stays set until the next reset request overwrites it, which is
    // fine: it is only read inside the reset handler.
    public bool IsRandomStartRequested() {
        return IsReady && _accessor.ReadByte(RANDOM_START_OFFSET) == 1;
    }

    // returns false when no step is pending. opponentAction is null when
    // python did not send one (opponent then stands still).
    public bool TryReadStep(out float[] action, out float[] opponentAction) {
        action = null;
        opponentAction = null;
        if (!IsStepRequested()) return false;

        // flag seen -> make sure the payload reads are not served stale
        Thread.MemoryBarrier();
        action = ReadFloats(ACT_OFFSET, ACT_DIM);
        if (_accessor.ReadByte(OPP_VALID_OFFSET) == 1) {
            opponentAction = ReadFloats(OPP_ACT_OFFSET, ACT_DIM);
        }
        return true;
    }

    void WritePayload(float[] observation, float reward, bool done, int learnerTeam, int scorer, int playMode, float turnSeconds) {
        if (observation.Length != OBS_DIM) {
            GD.PrintErr($"Observation needs {OBS_DIM} values, got {observation.Length}");
            return;
        }
        for (int i = 0; i < OBS_DIM; i++) {
            _accessor.Write(OBS_OFFSET + i * 4, observation[i]);
        }
        _accessor.Write(REWARD_OFFSET, reward);
        _accessor.Write(DONE_OFFSET, (byte)(done ? 1 : 0));
        _accessor.Write(LEARNER_TEAM_OFFSET, (byte)learnerTeam);
        _accessor.Write(SCORER_OFFSET, (byte)scorer);
        _accessor.Write(PLAY_MODE_OFFSET, (byte)playMode);
        _accessor.Write(TURN_SECONDS_OFFSET, turnSeconds);
    }

    // sends everything back to python after a step, then releases python.
    // barrier makes sure python never sees the flag drop before the
    // payload is written (matters on ARM). turnSeconds is how long this turn
    // took to settle in game seconds, so python can watch the turn-length spread.
    // oobReset (v5) is 1 when this turn hit an out-of-bounds kickoff, so python
    // can skip the position shaping over the jump.
    public void SendStepResults(float[] observation, float reward, bool done, int learnerTeam, int scorer, int playMode, float turnSeconds, bool oobReset = false) {
        if (!IsReady) return;
        WritePayload(observation, reward, done, learnerTeam, scorer, playMode, turnSeconds);
        _accessor.Write(OOB_RESET_OFFSET, (byte)(oobReset ? 1 : 0));
        Thread.MemoryBarrier();
        _accessor.Write(STEP_REQ_OFFSET, (byte)0);
    }

    // sends initial state after reset (a reset is not a turn, so 0 seconds, and
    // the oob flag is cleared so a fresh episode never inherits the last one's)
    public void SendResetResults(float[] observation, int learnerTeam, int playMode) {
        if (!IsReady) return;
        WritePayload(observation, 0f, false, learnerTeam, 0, playMode, 0f);
        _accessor.Write(OOB_RESET_OFFSET, (byte)0);
        Thread.MemoryBarrier();
        _accessor.Write(RESET_REQ_OFFSET, (byte)0);
    }

    // cleanup when we're done. the file is left on disk on purpose.
    public void Dispose() {
        try {
            _accessor?.Dispose();
            _mmf?.Dispose();
            _file?.Dispose();
        } catch (Exception e) { // catching them errors
            GD.PrintErr($"Error during cleanup: {e.Message}");
        } finally {
            _accessor = null;
            _mmf = null;
            _file = null;
        }
    }
}
