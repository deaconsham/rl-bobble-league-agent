using Godot;
using System;
using System.IO;

public partial class DataManager : Node {

    SharedMemory memory;

    [Export] Node players;
    [Export] RigidBody3D ball;

    const int PLAYERS_PER_TEAM = 3;
    const int OBS_DIM = 2 + PLAYERS_PER_TEAM * 2 * 2; // ball + 2 teams = 14

    // ---- learner frame ----
    // Team 1 attacks -Z in world space. Team 2's view is mirrored on Z
    // (x, z) -> (x, -z), so EVERY team sees itself attacking -Z.
    // One policy can then play either side. The spawns are mirror images
    // on Z, so kickoff looks identical from both sides.
    public static float FrameZ(int team) => team == 2 ? -1f : 1f;

    // layout: [ball_x, ball_z, f1_x, f1_z, f2_x, f2_z, f3_x, f3_z, e1_x, e1_z, e2_x, e2_z, e3_x, e3_z]
    // (friendly = learnerTeam, all in learnerTeam's frame)
    public void Export(bool done, float reward, bool reset, int learnerTeam, int scorer, int playMode) {
        if (memory == null || !memory.IsReady) return;
        float[] obs = BuildObservation(learnerTeam);

        if (reset) {
            memory.SendResetResults(obs, learnerTeam, playMode);
        } else {
            memory.SendStepResults(obs, reward, done, learnerTeam, scorer, playMode);
        }
    }

    float[] BuildObservation(int learnerTeam) {
        int enemyTeam = learnerTeam == 1 ? 2 : 1;
        float zSign = FrameZ(learnerTeam);

        float[] obs = new float[OBS_DIM];
        obs[0] = ball.GlobalPosition.X;
        obs[1] = ball.GlobalPosition.Z * zSign;
        WriteTeam(obs, 2, learnerTeam, zSign);
        WriteTeam(obs, 2 + PLAYERS_PER_TEAM * 2, enemyTeam, zSign);
        return obs;
    }

    // walks EVERY child, keeps this team's players in scene-tree order.
    // GameManager.TeamPlayers uses the same order, so action i <-> obs slot i.
    void WriteTeam(float[] obs, int start, int team, float zSign) {
        int found = 0;
        foreach (Node child in players.GetChildren()) {
            if (child.IsQueuedForDeletion()) continue;
            if (child is not Player player || player.team != team) continue;
            if (found >= PLAYERS_PER_TEAM) {
                GD.PrintErr($"Team {team} has more than {PLAYERS_PER_TEAM} players; extra ignored");
                break;
            }
            obs[start + found * 2] = player.GlobalPosition.X;
            obs[start + found * 2 + 1] = player.GlobalPosition.Z * zSign;
            found++;
        }
        if (found < PLAYERS_PER_TEAM) {
            GD.PrintErr($"Team {team} has only {found} players; missing slots sent as 0");
        }
    }

    public bool IsResetRequested() {
        return memory != null && memory.IsResetRequested();
    }

    public bool TryReadStep(out float[] action, out float[] opponentAction) {
        if (memory == null) {
            action = null;
            opponentAction = null;
            return false;
        }
        return memory.TryReadStep(out action, out opponentAction);
    }

    public override void _EnterTree() {
        memory = new SharedMemory();
    }

    public override void _ExitTree() {
        memory?.Dispose();
        memory = null;
    }

}
