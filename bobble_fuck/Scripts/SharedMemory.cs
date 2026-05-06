using Godot;
using System;
using System.IO.MemoryMappedFiles;

// bridge between godot and python for RL training
public class SharedMemory {
    private const int OBS_OFFSET = 0;
    private const int OBS_SIZE = 56;
    private const int ACT_OFFSET = 56;
    private const int ACT_SIZE = 24;
    private const int REWARD_OFFSET = 80;
    private const int DONE_OFFSET = 84;
    private const int STEP_REQ_OFFSET = 85;
    private const int RESET_REQ_OFFSET = 86;
    private const int TOTAL_SIZE = 87;

    private const int OBS_DIM = 14;
    private const int ACT_DIM = 6;

    private MemoryMappedFile _mmf;
    private MemoryMappedViewAccessor _accessor;
    
    public string SharedMemoryName = "BobbleFuckState";
    public bool CreateSharedMemory = true;

    public SharedMemory() {
        InitializeSharedMemory();
    }
    
    // sets up the shared memory segment
    private void InitializeSharedMemory() {
        try {
            if (CreateSharedMemory) {
                _mmf = MemoryMappedFile.CreateOrOpen(SharedMemoryName, TOTAL_SIZE);
                _accessor = _mmf.CreateViewAccessor(0, TOTAL_SIZE);
                
                for (int i = 0; i < TOTAL_SIZE; i++) {
                    _accessor.Write(i, (byte)0);
                }
                
                GD.Print($"Created shared memory '{SharedMemoryName}' ({TOTAL_SIZE} bytes)");
            } else {
                _mmf = MemoryMappedFile.OpenExisting(SharedMemoryName);
                _accessor = _mmf.CreateViewAccessor(0, TOTAL_SIZE);
                GD.Print($"Attached to shared memory '{SharedMemoryName}'");
            }
        } catch (Exception e) {
            GD.PrintErr($"Failed to init shared memory: {e.Message}");
        }
    }

    // writes game state to memory for python to read
    public void WriteObservation(float[] observation) {
        if (observation.Length != OBS_DIM) {
            GD.PrintErr($"Observation needs {OBS_DIM} values, got {observation.Length}");
            return;
        }

        for (int i = 0; i < OBS_DIM; i++) {
            _accessor.Write(OBS_OFFSET + i * 4, observation[i]);
        }
    }

    // gets the action from python
    public float[] ReadAction() {
        float[] action = new float[ACT_DIM];
        
        for (int i = 0; i < ACT_DIM; i++) {
            action[i] = _accessor.ReadSingle(ACT_OFFSET + i * 4);
        }
        
        return action;
    }

    // sends reward to python
    public void WriteReward(float reward) {
        _accessor.Write(REWARD_OFFSET, reward);
    }

    // tells python if episode ended
    public void WriteDone(bool done) {
        _accessor.Write(DONE_OFFSET, (byte)(done ? 1 : 0));
    }

    // checks if python wants us to step
    public bool IsStepRequested() {
        return _accessor.ReadByte(STEP_REQ_OFFSET) == 1;
    }

    // clears step flag after done
    public void ClearStepRequest() {
        _accessor.Write(STEP_REQ_OFFSET, (byte)0);
    }

    // checks if python wants reset
    public bool IsResetRequested() {
        return _accessor.ReadByte(RESET_REQ_OFFSET) == 1;
    }

    // clears the reset flag after done
    public void ClearResetRequest() {
        _accessor.Write(RESET_REQ_OFFSET, (byte)0);
    }

    // checks what python wants and returns action if stepping
    public float[] ProcessRequests(out bool resetRequested) {
        resetRequested = IsResetRequested();
        
        if (resetRequested) {
            return null;
        }

        if (IsStepRequested()) {
            return ReadAction();
        }

        return null;
    }

    // sends everything back to python after a step
    public void SendStepResults(float[] observation, float reward, bool done) {
        WriteObservation(observation);
        WriteReward(reward);
        WriteDone(done);
        ClearStepRequest();
    }

    // sends initial state after reset
    public void SendResetResults(float[] observation) {
        WriteObservation(observation);
        WriteReward(0.0f);
        WriteDone(false);
        ClearResetRequest();
    }

    ~SharedMemory () {
        CleanUp();
    }

    // cleanup when we're done
    private void CleanUp() {
        try {
            _accessor?.Dispose();
            _mmf?.Dispose();
        } catch (Exception e) { // catching them errors
            GD.PrintErr($"Error during cleanup: {e.Message}");
        }
    }
}
